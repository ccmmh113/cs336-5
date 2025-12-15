import os
import json
import argparse
from typing import List, Callable, Dict
# from vllm import LLM, SamplingParams
# from drgrpo_grader import r1_zero_reward_fn

# def evaluate_vllm(
#     vllm_model: LLM,
#     reward_fn: Callable[[str, str], Dict[str, float]],
#     prompts: List[str],
#     ground_truths: List[str],
#     eval_sampling_params: SamplingParams,
#     output_file: str
# ) -> None:
#     """
#     使用 vLLM 评估模型，计算奖励并序列化结果。
#     """
  
#     # 1. 批量生成
#     outputs = vllm_model.generate(prompts, eval_sampling_params)
    
#     results = []
#     stats = {
#         "total": 0,
#         "correct": 0,
#         "format_error": 0,
#         "correct_format_wrong_answer": 0
#     }

   
#     # 2. 计算 Reward 并统计
#     for output_item, gold_answer in zip(outputs, ground_truths):

#         prompt = output_item.prompt
#         # vLLM 输出
#         generated_text = output_item.outputs[0].text
        
#         # 调用评分函数 (此时 gold_answer 是纯数字字符串，如 "18")
#         scores = reward_fn(generated_text, gold_answer)
        
#         # 统计数据
#         stats["total"] += 1
#         if scores.get("reward", 0) == 1:
#             stats["correct"] += 1
        
#         if scores.get("format_reward", 0) == 0:
#             stats["format_error"] += 1
#         elif scores.get("answer_reward", 0) == 0:
#             stats["correct_format_wrong_answer"] += 1

#         # 序列化结果
#         result_entry = {
#             "prompt": prompt,
#             "ground_truth": gold_answer,
#             "generated_text": generated_text,
#             "scores": scores
#         }
#         results.append(result_entry)

#     # 3. 写入文件
#     os.makedirs(os.path.dirname(output_file), exist_ok=True)
#     with open(output_file, 'w', encoding='utf-8') as f:
#         for res in results:
#             f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
#     # 打印统计结果
#     accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
#     print("\n" + "="*40)
#     print(f"Evaluation Complete.")
#     print(f"Output saved to: {output_file}")
#     print(f"Total Examples: {stats['total']}")
#     print(f"Accuracy: {accuracy:.2%}")
#     print(f"Format Errors (Format=0): {stats['format_error']}")
#     print(f"Format OK, Answer Wrong (Format=1, Ans=0): {stats['correct_format_wrong_answer']}")
#     print(f"Fully Correct (Reward=1): {stats['correct']}")
#     print("="*40)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", type=str, default="model/Qwen2.5-Math-1.5B")
#     parser.add_argument("--data_path", type=str, default="data/gsm8k/zero_evaluate.jsonl")
#     parser.add_argument("--output_file", type=str, default="result/zero_shot_results_qwen1.5B.jsonl")
#     args = parser.parse_args()

#     # 1. 加载转换后的数据
#     print(f"加载数据 {args.data_path}...")
#     prompts = []
#     ground_truths = []
#     with open(args.data_path, "r", encoding="utf-8") as f:
#         for line in f:
#             item = json.loads(line)
#             prompts.append(item["prompt"])
#             ground_truths.append(item["ground_truth"])

#     # 2. 初始化 vLLM
#     print(f"Initializing vLLM with model: {args.model_path}")
#     llm = LLM(
#         model=args.model_path, # 根据你的 vLLM 版本，可能是 model 或者是 model_id
#         trust_remote_code=True,
#         tensor_parallel_size=1,
#         gpu_memory_utilization=0.7 # 调高利用率以获得更好性能
#     )

#     # 3. 设置采样参数 (按照 R1-Zero 评估标准)
#     sampling_params = SamplingParams(
#         temperature=1.0,
#         top_p=1.0,
#         max_tokens=1024,
#         stop=["</answer>"], # 生成到闭合标签停止
#         include_stop_str_in_output=True # 必须包含标签，否则 grader 无法解析格式
#     )

#     # 4. 运行评估
#     evaluate_vllm(
#         vllm_model=llm,
#         reward_fn=r1_zero_reward_fn,
#         prompts=prompts,
#         ground_truths=ground_truths,
#         eval_sampling_params=sampling_params,
#         output_file=args.output_file
#     )


def eval():
    correct = 0
    format_correct_wrong_ans = 0
    format_error = 0
    total = 0
    with open("data/zero_shot_results_qwen1.5B.jsonl", "r",encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            total += 1
            scores = data["scores"]
        
            if scores["reward"] == 1:
                correct += 1
            elif scores["format_reward"] == 1:
                format_correct_wrong_ans += 1
            else:
                format_error += 1

    print(f"📊 总样本: {total}")
    print(f"✅ 完全正确: {correct} ({correct/total:.2%})")
    print(f"⚠️ 格式对但答案错: {format_correct_wrong_ans} ({format_correct_wrong_ans/total:.2%})")
    print(f"❌ 格式错误: {format_error} ({format_error/total:.2%})")
if __name__ == "__main__":
    # main()
    eval()