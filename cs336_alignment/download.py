import json
import os
from datasets import load_dataset

PROMPT_PATH = "cs336_alignment/prompts/r1_zero.prompt"


def prepare_sft():
    dataset_name = "chenyn66/still_math_r1"
    save_path = "data/gsm8k/sft.jsonl"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"正在下载并转换 {dataset_name}...")
    ds = load_dataset(dataset_name, split="train")
    
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        r1_template = f.read().strip()

    count = 0
    with open(save_path, "w", encoding="utf-8") as f:
        for item in ds:
            # 1. 提取题目
            flag=item.get("label","")
            if flag ==True:
                problem = item.get("question", "").strip()
                # 2. 提取并s清洗思考过程
                # 原始 response 里带了 <think> 和 </think>，我们需要把内容抠出来
                raw_response = item.get("response", "").strip()
                if "<think>" in  raw_response and '</think>' in raw_response:
                    # 提取 <think> 标签中间的内容
                    thought = raw_response.split("<think>")[1].strip().split("</think>")[0].strip()
                else:
                    thought =raw_response.strip()
                    
                # 3. 提取答案 (直接用 ground_truth 字段最稳)
                answer = item.get("ground_truth", "").strip()

                if not thought or not answer:
                    continue

                # 4. 构造 Prompt
                full_prompt = r1_template.format(question=problem)
                
                # 5. 构造符合你 reward_fn 要求的 Response
                # 注意：这里我们手动补上中间的空格和 <answer> 标签
                full_response = f"<think> {thought} </think> <answer> {answer} </answer>"
                # print(full_prompt,full_response)
                json.dump({"prompt": full_prompt,"response": full_response}, f, ensure_ascii=False)
                f.write("\n")
                count += 1
        
    print(f"✅ 成功转换 {count} 条数据！文件保存在: {save_path}")

if __name__ == "__main__":
    # prepare_sft()
    with open("data/gsm8k/sft.jsonl", "r", encoding="utf-8") as f:
        for  _ in range(3):
            first_line = json.loads(f.readline())
            print(first_line['response'])