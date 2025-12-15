import torch
import json
import random
import wandb
import os
import argparse
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from unittest.mock import patch



# --- 导入自定义工具函数 ---
from cs336_alignment.utils import (
    tokenize_prompt_and_output, 
    get_response_log_probs,
)
from cs336_alignment.grpo_utils import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
    log_generations
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn

# ==========================================
# 辅助函数
# ==========================================
def load_math12k_dataset(path, prompt_template=None):
    df = pd.read_parquet(path)
    processed_items = []
    for _, row in df.iterrows():
        q_text = row['problem']
        gold_answer = row['answer']
        
        if gold_answer:
            processed_items.append({
                "prompt": prompt_template.replace("{question}", q_text),
                "gold": gold_answer
            })
    return processed_items

def load_gsm8k_dataset(path, prompt_template=None):
    
    processed_items = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            q_text = item['question']
            full_sol = item['answer']
            gold_answer = full_sol.split("####")[-1].strip() if "####" in full_sol else full_sol.strip()
            processed_items.append({
                "prompt": prompt_template.replace("{question}", q_text),
                "gold": gold_answer
            })
    return processed_items


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


# ==========================================
# GRPO 核心训练逻辑
# ==========================================

def run_grpo_training(args):
    # 1. 实验配置与初始化
    assert args.train_batch_size % args.gradient_accumulation_steps == 0
    micro_train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    
    assert args.rollout_batch_size % args.group_size == 0
    
    n_prompts_per_rollout = args.rollout_batch_size // args.group_size
    
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    
    if args.prompt_style == "question_only":
        active_reward_fn = question_only_reward_fn
    else:
        active_reward_fn = r1_zero_reward_fn

    print(f"Loading prompt template from {args.prompt_path}...")
    with open(args.prompt_path, "r") as f:
        prompt_template = f.read().strip()

    if 'math12k' in args.train_data_path.lower():
        questions_pool = load_math12k_dataset(args.train_data_path, prompt_template)
        val_samples = load_math12k_dataset(args.test_data_path, prompt_template)[:args.max_eval_samples]
    elif 'gsm8k' in args.train_data_path.lower():
        questions_pool = load_gsm8k_dataset(args.train_data_path, prompt_template)
        val_samples = load_gsm8k_dataset(args.test_data_path, prompt_template)[:args.max_eval_samples]
    else:
        raise ValueError("Unsupported dataset. Please use Math12K or GSM8K.")
    print(f"Total training questions: {len(questions_pool)}")

    # 定义评估用的采样参数 
    eval_sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # 3. 初始化模型
    print(f"Initializing Policy Model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2"
    ).to(args.device)
    # 开启梯度检查点
    policy.gradient_checkpointing_enable()
    
    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)

    print(f"Initializing vLLM on {args.vllm_device}...")
    vllm_inst = init_vllm(args.model_id, args.vllm_device, args.seed, args.vllm_gpu_util)


    # ==========================================
    # Step 0 初始评估 (训练前的基准)
    # ==========================================
    print(f"\n[GRPO Step 0] Starting Initial Evaluation (Baseline)...")
    policy.eval() 
    # 第一次同步初始权重
    load_policy_into_vllm_instance(policy, vllm_inst)
    
    metrics = log_generations(
        vllm_model=vllm_inst,
        sampling_params=eval_sampling_params,
        prompts=[s['prompt'] for s in val_samples],
        ground_truths=[s['gold'] for s in val_samples],
        reward_fn=active_reward_fn ,
        step=0, # 显式记为第 0 步
        log_prefix="eval"
    )
    print(f"[GRPO Step 0] Initial Eval Accuracy: {metrics.get('eval/accuracy', 0):.2%}")
    policy.train() 

    # ==========================================
    # 开始 GRPO 主循环
    # ==========================================
    # 4. GRPO 主循环
    global_step = 0
    # 训练采样参数
    rollout_sampling_params = SamplingParams(
        n=args.group_size,
        temperature=args.sampling_temperature,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    progress_bar = tqdm(range(args.n_grpo_steps), desc="GRPO Steps")
    for step in range(args.n_grpo_steps):
        policy.eval()
        load_policy_into_vllm_instance(policy, vllm_inst)  
        current_batch_questions = random.sample(questions_pool, n_prompts_per_rollout)
        prompts = [q['prompt'] for q in current_batch_questions]
        golds = [q['gold'] for q in current_batch_questions]     
        # vLLM 生成
        outputs = vllm_inst.generate(prompts, rollout_sampling_params)   
        flat_prompts = []
        flat_responses = []
        flat_golds = []
        for i, output in enumerate(outputs):
            for candidate in output.outputs:
                flat_prompts.append(prompts[i])
                flat_responses.append(candidate.text)
                flat_golds.append(golds[i])

        # 计算 Advantage
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=active_reward_fn ,
            rollout_responses=flat_responses,
            repeated_ground_truths=flat_golds,
            group_size=args.group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.use_std_normalization
        )
        
        wandb.log({
            "rollout/mean_reward": reward_meta['mean_reward'],
            "rollout/max_reward": reward_meta['max_reward'],
            "rollout_step": step + 1 # 显式 X 轴
        }, step=step + 1)

        # ==========================================
        # Phase 2: 准备训练数据 (过滤与重采样)
        # ==========================================
        print(f">> 正在进行长度过滤 (Max: {args.max_len})...")
        
        # 1. 过滤：获取长度合格的原始索引
        valid_indices = [
            i for i, (p, r) in enumerate(zip(flat_prompts, flat_responses))
            if len(tokenizer.encode(p + r, add_special_tokens=False)) <= args.max_len
        ]
        
        num_valid = len(valid_indices)
        if num_valid == 0:
            print(f"⚠️ 警告: 当前批次所有样本均超过 {args.max_len}，跳过训练。")
            continue
            
        # 2. 重采样补齐：确保数据量达到训练要求的 Batch Size
        if num_valid < args.train_batch_size:
            gap = args.train_batch_size - num_valid
            print(f"⚠️ 有效样本({num_valid})不足，随机补齐缺口({gap})...")
            # 允许重复采样有效索引
            extra_indices = np.random.choice(valid_indices, size=gap, replace=True)
            final_indices = valid_indices + extra_indices.tolist()
        else:
            final_indices = valid_indices

        # 3. 同步更新文本列表与奖励张量
        filtered_prompts = [flat_prompts[i] for i in final_indices]
        filtered_responses = [flat_responses[i] for i in final_indices]
        
        # 必须先基于索引对 Tensor 进行切片，再移动到 GPU
        # 注意：此处切片会自动处理重采样导致的重复行
        advantages = advantages[final_indices].to(args.device)
        raw_rewards = raw_rewards[final_indices].to(args.device)

        # 4. 张量化 (Tokenization)
        all_inputs    = tokenize_prompt_and_output(filtered_prompts, filtered_responses, tokenizer)
        all_input_ids = all_inputs['input_ids'].to(args.device)
        all_labels    = all_inputs['labels'].to(args.device)
        all_masks     = all_inputs['response_mask'].to(args.device)
        
        print(f">> 处理完成: 采样 {len(flat_responses)} -> 有效训练样本 {all_input_ids.size(0)}")

        # 5. 计算 Old Log Probs (必须在补齐后进行，且精度需对齐)
        policy.eval()
        with torch.no_grad(): # 关键：确保精度与训练阶段(BF16)完全一致
            old_log_probs_list = []
            for i in range(0, all_input_ids.size(0), micro_train_batch_size):
                batch_ids = all_input_ids[i : i + micro_train_batch_size]
                batch_lbls = all_labels[i : i + micro_train_batch_size]
                res = get_response_log_probs(policy, batch_ids, batch_lbls)
                old_log_probs_list.append(res['log_probs'])
            old_log_probs = torch.cat(old_log_probs_list, dim=0)

        del old_log_probs_list # 显式删除不再需要的中间大列表
        torch.cuda.empty_cache() # 确保进入 Phase 3 训练时显存最空
        # ==========================================
        # Phase 3: 训练 
        # ==========================================
        actual_train_size = all_input_ids.size(0)
        num_updates_per_epoch = actual_train_size // args.train_batch_size
        
        if num_updates_per_epoch == 0:
            print("⚠️ 剩余样本不足一个训练 Batch，跳过此步更新。")
            continue

        policy.train()
        for epoch in range(args.epochs_per_rollout_batch):
            # 打乱实际训练样本的索引
            dataset_indices = np.random.permutation(actual_train_size)
            
            for update_step in range(num_updates_per_epoch):
                # 2. 锁定当前“逻辑 Batch”的索引范围
                logical_batch_start = update_step * args.train_batch_size
                logical_batch_end = (update_step + 1) * args.train_batch_size
                logical_indices = dataset_indices[logical_batch_start : logical_batch_end]
                
                if args.length_norm_type == "mask_dapo":
                    # DAPO 模式：分母是当前整个逻辑 Batch 的有效 Token 总数
                    # 此时内层函数不应再除以 accumulation_steps，因为它已经是在全局维度缩放了
                    norm_val = all_masks[logical_indices].sum().item()
                    acc_steps_to_pass = 1 
                elif args.length_norm_type == "mask_normalize":
                    # 固定常数 C (max_len)
                    norm_val = args.max_len
                    # 这种模式下通常需要除以梯度累积步数来平均
                    acc_steps_to_pass = args.gradient_accumulation_steps
                else: # mask_mean
                    # 默认模式：由内层函数处理每个样本的 Mean
                    norm_val = 1.0 # 这个值在 mask_mean 模式下通常不被使用
                    acc_steps_to_pass = args.gradient_accumulation_steps
                    
                # 清空上一逻辑步的梯度
                optimizer.zero_grad()
                
                # 用于累积监控指标
                batch_loss = 0
                batch_clip_frac = 0
                batch_avg_response_entropy = 0
                batch_avg_global_entropy = 0
                batch_ratio_mean = 0
                

                # 3. 开启“梯度累积循环” (内层循环)
                for micro_step in range(args.gradient_accumulation_steps):
                    # 计算当前物理 Micro-batch 在逻辑 Batch 里的切片位置
                    m_start = micro_step * micro_train_batch_size
                    m_end = (micro_step + 1) * micro_train_batch_size
                    micro_indices = logical_indices[m_start : m_end]
                    
                    # --- A. 准备 Micro-batch 数据 ---
                    mb_input_ids = all_input_ids[micro_indices]
                    mb_labels = all_labels[micro_indices]
                    mb_masks = all_masks[micro_indices]
                    mb_advs = advantages[micro_indices].unsqueeze(1) # 转为 [B, 1] 以便广播
                    mb_old_lps = old_log_probs[micro_indices]
                    mb_raw_r = raw_rewards[micro_indices].unsqueeze(1)

                    # --- B. 前向传播计算当前 Log-probs ---
                    res = get_response_log_probs(
                        policy, mb_input_ids, mb_labels, return_token_entropy=True
                    )
                    
                    # --- C. 执行单步微批次更新 (包含 backward) ---
                    # 注意：函数内部执行了 scaled_loss.backward()
                    # 梯度会累加在参数的 .grad 中
                    _, loss_meta = grpo_microbatch_train_step(
                        policy_log_probs=res['log_probs'],
                        response_mask=mb_masks,
                        gradient_accumulation_steps=acc_steps_to_pass,
                        loss_type=args.loss_type,
                        raw_rewards=mb_raw_r,
                        advantages=mb_advs,
                        old_log_probs=mb_old_lps,
                        cliprange=0.2,
                        length_norm_type=args.length_norm_type,
                        constant_normalizer = norm_val
                    )
                    
                    
                    with torch.no_grad():
                        token_entropy = res['token_entropy']
                        valid_token_mask = (mb_labels != tokenizer.pad_token_id)
                        current_res_mask = mb_masks.bool() & valid_token_mask
                        
                        avg_res_entropy = token_entropy[current_res_mask].mean().item() if current_res_mask.any() else 0.0
                        avg_global_entropy = token_entropy[valid_token_mask].mean().item()
                    
                    # 累加指标用于后续 WandB
                    batch_loss += loss_meta['loss'].item()
                    batch_clip_frac += loss_meta.get('clip_fraction', 0)
                    batch_avg_response_entropy += avg_res_entropy
                    batch_avg_global_entropy += avg_global_entropy
                    batch_ratio_mean += loss_meta.get('ratio_mean', 0)

                # 4. 一个逻辑 Batch 累积完成，执行权重更新
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                global_step += 1
                # --- 日志记录 ---
                if global_step % 5 == 0:
                    wandb.log({
                        "train/loss": batch_loss / args.gradient_accumulation_steps,
                        "train/clip_fraction": batch_clip_frac / args.gradient_accumulation_steps,
                        "train_step": global_step,
                        "train/grad_norm": grad_norm.item(),
                        "train/response_entropy": batch_avg_response_entropy / args.gradient_accumulation_steps,
                        "train/global_entropy": batch_avg_global_entropy / args.gradient_accumulation_steps,
                        "train/ratio_mean": batch_ratio_mean / args.gradient_accumulation_steps,
                    }, step=step + 1) 

        progress_bar.update(1)
        torch.cuda.empty_cache()

        # ==========================================
        # Phase 4: 评估与保存 
        # ==========================================
        
        # 1. 评估逻辑 (每隔 args.eval_every_steps 个 GRPO Step)
        if (step + 1) % args.eval_every_steps == 0:
            print(f"\n[GRPO Step {step + 1}] Starting Evaluation...")
            policy.eval() 
            load_policy_into_vllm_instance(policy, vllm_inst)
            
            metrics = log_generations(
                vllm_model=vllm_inst,
                sampling_params=eval_sampling_params,
                prompts=[s['prompt'] for s in val_samples],
                ground_truths=[s['gold'] for s in val_samples],
                reward_fn=active_reward_fn ,
                step=step + 1,
                log_prefix="eval"
            )
            print(f"[GRPO Step {step + 1}] Eval Accuracy: {metrics.get('eval/accuracy', 0):.2%}")
            policy.train()

        # 2. 保存逻辑
        if (step + 1) % args.save_every_steps == 0:
            print(f"Saving checkpoint at step {step + 1}...")
            save_path = os.path.join(args.output_dir, f"grpo_step{step+1}")
            os.makedirs(save_path, exist_ok=True)
            policy.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    print("GRPO Training Finished.")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 路径
    parser.add_argument("--model_id", type=str, default="model/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data_path", type=str, default="data/gsm8k/train.jsonl")
    parser.add_argument("--test_data_path", type=str, default="data/gsm8k/test.jsonl")
    
    parser.add_argument("--prompt_path", type=str, default="cs336_alignment/prompts/r1_zero.prompt")
    parser.add_argument("--output_dir", type=str, default="result/grpo_checkpoints")
    
    # 核心超参数
    parser.add_argument("--n_grpo_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--rollout_batch_size", type=int, default=256)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128)
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1)
    
    # 采样参数
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--sampling_max_tokens", type=int, default=1024)
    parser.add_argument("--sampling_min_tokens", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=2048)
    
    # GRPO 特定
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument("--use_std_normalization", action="store_true", help="Enable std normalization")
    parser.add_argument("--loss_type", type=str, default="grpo_clip")
    parser.add_argument("--length_norm_type", type=str, default="mask_mean")
    
    # 硬件与评估
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vllm_device", type=str, default="cuda:1")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every_steps", type=int, default=8)
    parser.add_argument("--save_every_steps", type=int, default=100)
    parser.add_argument("--max_eval_samples", type=int, default=2000)

    # Prompt与奖励函数
    parser.add_argument("--prompt_style", type=str, default="r1_zero", choices=["r1_zero", "question_only"], 
                    help="选择提示词风格和对应的奖励函数")
    
    # WandB
    parser.add_argument("--wandb_project", type=str, default="cs336-grpo")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()
    run_grpo_training(args)