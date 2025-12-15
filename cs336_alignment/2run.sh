#!/bin/bash

# --- 基础配置 ---
MODEL_PATH="model/Qwen2.5-Math-1.5B"
TRAIN_DATA="data/gsm8k/sft.jsonl"
VAL_DATA=  "data/gsm8k/test.jsonl"
LOG_DIR=   "logs/sft_experiments"
mkdir -p $LOG_DIR

# 确保 WandB 已登录
# wandb login <你的API_KEY>

echo "开始执行 CS336 SFT 对比实验..."

# 实验序列：不同数据量
for SIZE in 128 256 512 1024
do
    echo "=========================================="
    echo "正在运行实验: 数据量=${SIZE}"
    echo "=========================================="
    
    python your_script.py \
        --model_id $MODEL_PATH \
        --dataset_size $SIZE \
        --train_data_path $TRAIN_DATA \
        --val_data_path $VAL_DATA \
        --max_steps 200 \
        --eval_every_steps 20 \
        --device cuda:0 \
        --vllm_device cuda:1 \
        --vllm_gpu_util 0.4 \
        --wandb_run_name "qwen_sft_size_${SIZE}" > "$LOG_DIR/size_${SIZE}.log" 2>&1
done

# 实验序列：数据过滤 (Filter Correct)
echo "=========================================="
echo "正在运行实验: 过滤正确示例"
echo "=========================================="
python your_script.py \
    --model_id $MODEL_PATH \
    --filter_correct \
    --train_data_path $TRAIN_DATA \
    --val_data_path $VAL_DATA \
    --max_steps 300 \
    --eval_every_steps 20 \
    --device cuda:0 \
    --vllm_device cuda:1 \
    --vllm_gpu_util 0.4 \
    --wandb_run_name "qwen_sft_filtered" > "$LOG_DIR/filtered.log" 2>&1

echo "所有实验运行完毕！"

# 自动关机命令 (AutoDL 实例支持)
# shutdown