import torch
import json
import random
import wandb
import os
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          get_cosine_schedule_with_warmup)
from vllm import LLM, SamplingParams
from unittest.mock import patch

# --- 导入自定义工具函数 ---
from utils import (
    tokenize_prompt_and_output, 
    sft_microbatch_train_step,
    compute_entropy,
    log_generations
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def init_vllm(model_id, device, seed, gpu_memory_utilization):
    """初始化 vLLM 实例"""
    with patch("torch.distributed.get_world_size", return_value=1), \
         patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None):
        return LLM(
            model=model_id, 
            device=device, 
            dtype=torch.bfloat16,
            enable_prefix_caching=True, 
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed
        )

def load_policy_into_vllm_instance(policy, llm):
    """同步权重"""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
    print("\n[Sync] Policy weights synced to vLLM.")

def get_batch(tokenized_data, batch_size, device):
    """
    从预处理好的数据中随机采样一个 Batch。
    实现 Infinite Dataloader 的逻辑。
    """
    total_len = len(tokenized_data["input_ids"])
    batch_indices = random.sample(range(total_len), batch_size)
    
    return {
        "input_ids": tokenized_data["input_ids"][batch_indices].to(device),
        "labels": tokenized_data["labels"][batch_indices].to(device),
        "response_mask": tokenized_data["response_mask"][batch_indices].to(device)
    }




def run_expert_iteration(args):
    # 1. 基础配置
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    
    with open(args.prompt_path, "r") as f:
        r1_template = f.read().strip()

    # 2. 模型与分词器初始化
    print(f"Initializing Model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    ).to(args.device)


    policy.gradient_checkpointing_enable()
    
    optimizer = AdamW(policy.parameters(), lr=args.lr)


    total_train_steps = args.num_iterations * args.steps_per_iteration
    warm_up_steps = int(0.1 * total_train_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warm_steps=warm_up_steps,
        num_training_steps=total_train_steps
    )
    
    print(f"Initializing vLLM on {args.vllm_device}...")
    vllm_inst = init_vllm(args.model_id, args.vllm_device, args.seed, args.vllm_gpu_util)
    

    # ----------------------------------------------------------
    # 3. 提取训练集的 Prompts 和 Ground Truths (用于动态生成)
    # ----------------------------------------------------------
    print(f"加载训练集问题 {args.train_data_path}...")
    train_prompts, train_gts = [], []

    with open(args.train_data_path, "r",encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            raw_a = item.get('answer', '')
            gold = raw_a.split("####")[-1].strip() if "####" in raw_a else raw_a.strip()
            formatted_prompt = r1_template.replace("{question}", item.get('question', ''))
            train_prompts.append(formatted_prompt)
            train_gts.append(gold)
            
    if args.dataset_size:
        combined = list(zip(train_prompts, train_gts))
        combined = random.sample(combined, min(args.dataset_size, len(combined)))
        train_prompts, train_gts = zip(*combined)

    # 加载验证集 (保持不变)
    val_prompts, val_ground_truths = [], []
    with open(args.val_data_path, "r",encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.max_eval_samples: break
            item = json.loads(line)
            raw_a = item['answer']
            gold = raw_a.split("####")[-1].strip() if "####" in raw_a else raw_a.strip()
            formatted_prompt = r1_template.replace("{question}", item['question'])
            val_prompts.append(formatted_prompt)
            val_ground_truths.append(gold)

    eval_sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=args.max_tokens, 
        stop=["</answer>"], 
        include_stop_str_in_output=True
    )
    # 探索生成时的采样参数 (Temperature > 0 鼓励多样性，n 为每个 prompt 生成的条数)
    rollout_sampling_params = SamplingParams(
        temperature=args.rollout_temperature, 
        top_p=0.9, 
        n=args.samples_per_prompt, 
        max_tokens=args.max_tokens, 
        stop=["</answer>"], 
        include_stop_str_in_output=True
    )

    # ----------------------------------------------------------
    # 4. Expert Iteration 主循环
    # ----------------------------------------------------------
    global_step = 0
    for iteration in range(args.ei_epoch):
        print(f"\n========== [Iteration {iteration + 1}/{args.num_iterations}] ==========")
        
        # 4.1 权重同步到 vLLM，用于数据生成
        policy.eval()
        load_policy_into_vllm_instance(policy, vllm_inst)
        
        # 4.2 评估 
        metrics = log_generations(
            vllm_model=vllm_inst, sampling_params=eval_sampling_params,
            prompts=val_prompts, ground_truths=val_ground_truths,
            reward_fn=r1_zero_reward_fn, step=global_step, log_prefix="eval"
        )
        print(f"Iteration {iteration+1} Eval Accuracy: {metrics.get('eval/accuracy', 0):.2%}")

        # 4.3 生成探索数据 (Rollout)
        print("开始生成探索数据 (Rollout)...")
        rollout_outputs = vllm_inst.generate(train_prompts, rollout_sampling_params)
        
        # 4.4 评分与过滤 (Filtering)
        dynamic_train_data = []
        correct_count = 0
        total_generated = 0
        
        for i, req_output in enumerate(rollout_outputs):
            gt = train_gts[i]
            prompt_str = req_output.prompt
            
            for gen in req_output.outputs:
                total_generated += 1
                gen_text = gen.text
                reward = r1_zero_reward_fn(gen_text, gt) 
                if reward > 0: # 只保留生成正确的答案
                    dynamic_train_data.append({
                        "prompt": prompt_str,
                        "response": gen_text
                    })
                    correct_count += 1
                    
        print(f"生成完毕: 共生成 {total_generated} 条, 正确 {correct_count} 条 (正确率 {correct_count/total_generated:.2%})")
        
        if len(dynamic_train_data) == 0:
            print("警告: 本轮没有生成任何正确的答案，跳过训练步骤。")
            continue

        # 4.5 重新 Tokenize 动态数据集
        print("Tokenize 当前轮次的正确数据...")
        tokenized_train_data = tokenize_prompt_and_output(
            prompt_strs=[item['prompt'] for item in dynamic_train_data],
            output_strs=[item['response'] for item in dynamic_train_data],
            tokenizer=tokenizer
        )

        # 4.6 内层 SFT 训练
        policy.train()
        progress_bar = tqdm(range(args.per_steps), desc=f"SFT Steps (Iter {iteration+1})")
        grad_accum_steps = args.gradient_accumulation_steps
        for _ in range(args.per_steps):
            accumulated_loss = 0.0
            accumulated_entropy = 0.0
            accumulated_res_entropy = 0.0
            
            for _ in range(grad_accum_steps):
                batch = get_batch(tokenized_train_data, args.micro_batch_size, args.device)
                logits = policy(batch["input_ids"]).logits
                
                lse = torch.logsumexp(logits, dim=-1)
                target_logits = torch.gather(logits, -1, batch["labels"].unsqueeze(-1)).squeeze(-1)
                log_probs = target_logits - lse

                with torch.no_grad():
                    token_entropy = compute_entropy(logits)
                    valid_token_mask = (batch["labels"] != tokenizer.pad_token_id)
                    current_res_mask = batch["response_mask"].bool() & valid_token_mask
                    
                    avg_res_entropy = token_entropy[current_res_mask].mean().item() if current_res_mask.any() else 0.0
                    avg_global_entropy = token_entropy[valid_token_mask].mean().item()

                loss, _ = sft_microbatch_train_step(
                    policy_log_probs=log_probs,
                    response_mask=batch["response_mask"],
                    gradient_accumulation_steps=grad_accum_steps,
                    normalize_constant=1.0
                )
                
                accumulated_loss += loss.item() * grad_accum_steps
                accumulated_entropy += avg_global_entropy
                accumulated_res_entropy += avg_res_entropy

            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            progress_bar.update(1)
            
            wandb.log({
                "train/lr": scheduler.get_last_lr()[0],
                "train/loss": accumulated_loss / grad_accum_steps,
                "train/global_entropy": accumulated_entropy / grad_accum_steps,
                "train/response_entropy": accumulated_res_entropy / grad_accum_steps,
                "train_step": global_step,
                "iteration": iteration + 1
            })

    # 5. 保存模型
    print("Training finished. Saving model...")
    save_name = f"exit_iters{args.num_iterations}_subset{args.dataset_size}"
    output_dir = os.path.join(args.output_dir, save_name)
    os.makedirs(output_dir, exist_ok=True)
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS336 Expert Iteration Training")
    parser.add_argument("--model_id", type=str, default="model/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data_path", type=str, default="data/gsm8k/train.jsonl") # 注意这里最好换成带题目的原始train.jsonl
    parser.add_argument("--val_data_path", type=str, default="data/gsm8k/test.jsonl") 
    parser.add_argument("--prompt_path", type=str, default="cs336_alignment/prompts/r1_zero.prompt")
    parser.add_argument("--output_dir", type=str, default="result/checkpoints")
    
    # ExIt 新增参数 
    parser.add_argument("--ei_epoch", type=int, default=5, help="专家迭代的总轮数")
    parser.add_argument("--per_steps", type=int, default=50, help="每轮迭代中基于新生成数据训练的 Step 数")
    parser.add_argument("--samples_per_prompt", type=int, default=8, help="Rollout时每个问题生成的候选答案数量 (N)")
    parser.add_argument("--rollout_temperature", type=float, default=0.7, help="生成探索数据时的 Temperature")
    
    # 原本的训练参数
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--dataset_size", type=int, default=None)
    
    # 硬件与评估
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vllm_device", type=str, default="cuda:1")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.5)
    parser.add_argument("--max_eval_samples", type=int, default=100)
    
    # WandB
    parser.add_argument("--wandb_project", type=str, default="cs336-exit")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()
    run_expert_iteration(args)