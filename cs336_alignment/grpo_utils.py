import torch
from typing import Callable, List, Dict, Tuple, Literal, Optional
import torch
from typing import List, Dict, Callable
from transformers import PreTrainedTokenizer
import torch.nn.functional as F
from transformers import PreTrainedModel
import numpy as np
import wandb
from vllm import LLM, SamplingParams


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], Dict[str, float]],
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    eps: float = 1e-8,
    normalize_by_std: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """    
        rollout_responses (List[str]): 模型生成的所有回答列表。
            - 长度 = (问题数量 * group_size)。
            - 顺序隐含了分组逻辑，即前 group_size 个对应第1个问题，接下来的 group_size 个对应第2个问题...
        repeated_ground_truths (List[str]): 对应的标准答案列表。
            - 长度必须与 rollout_responses 相同。
            - 对于同一个问题的 group_size 个回答，这里的 ground_truth 是重复的。  
        group_size (int): 每个问题生成的回答数量 (G)。
            - 用于将扁平的列表 reshape 成 (N, G) 的矩阵。
        normalize_by_std (bool): 是否除以组内标准差。
            - True: 使用公式 (r - mean) / std (标准 GRPO/DeepSeekMath)。
            - False: 仅使用公式 r - mean (Dr. GRPO 建议的简化版，防止 std 接近 0 时数值不稳定)。

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
            1. advantages: 归一化后的优势分数，形状 (Total_Samples,)。 # Total_Samples = N * G
            2. raw_rewards: 原始奖励分数，形状 (Total_Samples,)。
            3. metadata: 包含统计信息（如平均奖励、最大/最小奖励等）的字典，用于日志记录。
    """
    
    # --- 1. 基础校验 ---
    # 确保输入数据长度一致，且能被 group_size 整除
    assert len(rollout_responses) == len(repeated_ground_truths), "Response 和 Ground Truth 数量必须一致"
    assert len(rollout_responses) % group_size == 0, "总样本数必须是 group_size 的整数倍"

    # --- 2. 计算原始奖励 (Raw Rewards) ---
    raw_rewards_list = []
    
    # 遍历每一对 (回答, 标答)
    for response, truth in zip(rollout_responses, repeated_ground_truths):
        # 调用外部传入的奖励函数
        # reward_fn 返回示例: {"reward": 1.0, "format_reward": 1.0, ...}
        score_dict = reward_fn(response, truth)
        
        # 我们只取最终的总分 "reward"
        raw_rewards_list.append(score_dict["reward"])
    
    # 将列表转换为 PyTorch 张量，形状为 (Total_Samples,)，例如 (32,)
    # device 默认为 CPU，后续训练时会自动移到 GPU
    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)

    # --- 3. 组内统计 (Group Statistics) ---
    # 计算有多少个独立的问题 (N)
    num_questions = raw_rewards.shape[0] // group_size
    
    # 关键步骤：Reshape
    # 将一维张量 (N*G, ) 变成二维张量 (N, G)
    # 每一行代表一个问题对应的 G 个回答的分数
    grouped_rewards = raw_rewards.view(num_questions, group_size)
    
    # 计算组内均值
    # dim=1 表示沿着“组”的维度（列）求平均
    # keepdim=True 保持形状为 (N, 1)，这是为了后面能利用广播机制让 (N, G) 减去 (N, 1)
    group_means = grouped_rewards.mean(dim=1, keepdim=True)
        
    # --- 4. 计算优势 (Advantage Calculation) ---
    if normalize_by_std:
        group_stds = grouped_rewards.std(dim=1, keepdim=True)
        # 对应公式 (28)
        # 广播机制：(N, G) - (N, 1) -> (N, G)
        # 每个分数减去它所在组的平均分，再除以标准差
        advantages = (grouped_rewards - group_means) / (group_stds + eps)
    else:
        # 对应公式 (31) - Dr. GRPO 的建议
        # 仅减去均值
        advantages = grouped_rewards - group_means

    # --- 5. 还原形状 ---
    # 将二维的优势矩阵 (N, G) 拉平回一维 (N*G, )，以便和 log_probs 的形状对齐
    advantages = advantages.view(-1)

    # --- 6. 生成元数据 (Metadata) ---
    # 这些数据不参与计算图，仅用于 WandB/日志 监控训练进度
    metadata = {
        "mean_reward": raw_rewards.mean().item(), # 整体平均分
        "std_reward": raw_rewards.std().item(),   # 整体标准差
        "max_reward": raw_rewards.max().item(),   # 最高分
        "min_reward": raw_rewards.min().item(),   # 最低分
        "mean_advantage": advantages.mean().item(), # 平均优势（理论上应接近0）
    }

    return advantages, metadata



def compute_naive_policy_gradient_loss(
    advantages: torch.Tensor,
    log_probs: torch.Tensor,
    mask: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    朴素策略梯度损失计算。
    """
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)
    # 形状(batch_size, 1)
    # 1. 计算逐 Token 损失
    per_token_loss = -(advantages * log_probs)
    
    # 2. 应用掩码求平均
    # masked_loss = per_token_loss * mask
    # loss = masked_loss.sum() / (mask.sum() + 1e-8)
    loss=per_token_loss
    # 3. 监控指标
    with torch.no_grad():
        metadata = {
            "pg_loss": loss.item(),
            "avg_log_probs": (log_probs * mask).sum().item() / (mask.sum().item() + 1e-8),
            "entropy_estimate": -(log_probs * torch.exp(log_probs) * mask).sum().item() / (mask.sum().item() + 1e-8)
        }
        
    return loss, metadata

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    计算 GRPO-Clip 损失函数。
    
    该函数实现了 PPO/GRPO 的核心安全机制：通过限制新旧策略的比率(Ratio)，
    防止模型在单次更新中变化过大，从而保证训练的稳定性。

    Args:
        advantages: 组内归一化后的优势值，形状 (batch_size, 1)。
        policy_log_probs: 当前正在优化的策略的对数概率，形状 (batch_size, sequence_length)。
        old_log_probs: 采样时旧策略的对数概率，形状 (batch_size, sequence_length)。
        cliprange: 截断阈值 epsilon (如 0.2)，定义了 [1-eps, 1+eps] 的“安全区”。

    Returns:
        loss: 逐 token 的损失标量，用于反向传播。
        metadata: 包含 clip_fraction (触发截断的频率) 等监控指标。
    """
    
    # 1. 计算概率比率 ratio = exp(log_prob_new - log_prob_old)
    # 使用对数空间相减再取指数，是为了保证在处理极小概率时的数值稳定性，防止下溢出。
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    
    # 2. 计算未截断的目标项 (Surrogate 1): 原始策略梯度
    # 如果不做限制，模型会沿着这个方向无限优化，导致策略崩溃。
    surr1 = ratio * advantages
    
    # 3. 计算截断后的目标项 (Surrogate 2): 限制更新幅度
    # 将比率强制约束在 [1-eps, 1+eps] 范围内。
    ratio_clipped = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    surr2 = ratio_clipped * advantages
    
    # 4. 取最小值 (min) 并取负 (实现梯度上升)
    # 核心逻辑：min(surr1, surr2) 建立了一个“悲观下界”。
    # - 当 A > 0 (奖励): 限制概率增加的上限。若 ratio > 1+eps，min 会选 surr2，此时梯度为 0。
    # - 当 A < 0 (惩罚): 限制概率减少的下限。若 ratio < 1-eps，min 会选 surr2，此时梯度为 0。
    # 只有当模型在安全区内更新，或者模型在“纠正错误方向”的更新时，才会选择 surr1 并保留梯度。
    loss = -torch.min(surr1, surr2)
    
    # 5. 计算元数据 (Metadata) 用于训练监控
    with torch.no_grad():
  
        clipped_mask = ( surr2< surr1).float()
        clip_fraction = clipped_mask.mean()
        
        metadata = {
            "clip_fraction": clip_fraction, # 非常关键：若此值接近 1.0，说明学习率过高或模型已停止学习
            "ratio_mean": ratio.mean(),
            "ratio_max": ratio.max(),
            "ratio_min": ratio.min(),
        }
        
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    old_log_probs: Optional[torch.Tensor] = None,
    cliprange: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    策略梯度损失的统一包装器。根据 loss_type 分发到不同的计算逻辑。

    Args:
        policy_log_probs: 当前策略的对数概率，形状 (batch_size, sequence_length)。
        loss_type: 损失类型字符串。
        raw_rewards: 原始奖励，no_baseline 模式必填。形状 (batch_size, 1)。
        advantages: 归一化优势，reinforce_with_baseline 和 grpo_clip 模式必填。形状 (batch_size, 1)。
        old_log_probs: 旧策略对数概率，grpo_clip 模式必填。形状 (batch_size, sequence_length)。
        cliprange: 截断参数 epsilon，grpo_clip 模式必填。

    Returns:
        Tuple[loss, metadata]: 逐 token 损失张量及元数据字典。
    """

    metadata = {}

    if loss_type == "no_baseline":
        # 断言检查：确保传入了必要的原始奖励
        assert raw_rewards is not None, "no_baseline 模式必须提供 raw_rewards"
        
        # 使用朴素公式：Loss = -raw_reward * log_prob
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs
        )

    elif loss_type == "reinforce_with_baseline":
        # 断言检查：确保传入了归一化后的优势值
        assert advantages is not None, "reinforce_with_baseline 模式必须提供 advantages"
        
        # 使用朴素公式：Loss = -advantage * log_prob
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs
        )

    elif loss_type == "grpo_clip":
        # 断言检查：GRPO 需要比率计算所需的 old_log_probs 和 clip 参数
        assert advantages is not None, "grpo_clip 模式必须提供 advantages"
        assert old_log_probs is not None, "grpo_clip 模式必须提供 old_log_probs"
        assert cliprange is not None, "grpo_clip 模式必须提供 cliprange"
        
        # 调用复杂的 GRPO 截断公式
        loss, grpo_metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange
        )
        metadata.update(grpo_metadata)
    else:
        raise ValueError(f"不支持的 loss_type: {loss_type}")

    return loss, metadata


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    在尊重布尔掩码的情况下，对张量元素求和并按常数归一化。
    
    Args:
        tensor: 需要求和的张量（例如每位置的 log_probs）。
        mask: 与 tensor 形状相同的掩码，1 表示包含，0 表示排除。
        normalize_constant: 归一化常数（除数）。
        dim: 沿着哪个维度求和。如果为 None，则对所有元素求和。
        
    Returns:
        归一化后的求和结果。
    """
    # 将 tensor 与 mask 相乘，排除 mask == 0 的元素
    masked_tensor = tensor * mask
    
    total_sum = torch.sum(masked_tensor, dim=dim)
        
    #  除以归一化常数
    return total_sum / normalize_constant

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
) -> torch.Tensor:
    """
    在尊重掩码的情况下计算张量的平均值。
    如果某个维度上的有效元素个数为 0，结果应为 NaN。

    Args:
        tensor: 需要计算平均值的数据张量。
        mask: 掩码张量，1 表示有效，0 表示无效。
        dim: 执行平均操作的维度。
    """
    # 1. 确保掩码与数据张量的类型一致
    mask = mask.to(tensor.dtype)

    # 2. 计算有效元素的总和 (tensor * mask 将无效位置清零)
    masked_sum = torch.sum(tensor * mask, dim=dim)

    # 3. 计算有效元素的数量（分母）
    mask_count = torch.sum(mask, dim=dim)
    
    return masked_sum / mask_count

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    old_log_probs: Optional[torch.Tensor] = None,
    cliprange: Optional[float] = None,
    constant_normalizer: Optional[float] = None,
    length_norm_type="mask_mean" # "mask_mean", "mask_normalize", "mask_dapo"
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # 1. 计算 Per-token Loss (Batch, Seq)
    per_token_loss, loss_metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange
    )
    
    # 2. 聚合 Loss (Aggregation)
    if length_norm_type == "mask_dapo":
        # --- DAPO 逻辑：全局 Token 平均 ---
        # 直接计算当前 micro-batch 的所有有效 token 的 loss 总和
        # 然后除以【外部传入的逻辑 Batch Token 总数】
        # 注意：这里不需要再除以 microbatch_size 或 gradient_accumulation_steps
        # 因为 constant_normalizer 已经是全局分母了
        total_masked_loss = (per_token_loss * response_mask).sum()
        scaled_loss = total_masked_loss / (constant_normalizer + 1e-8)
        
        # 为了方便日志观察，我们转回一个类似于平均 loss 的值
        microbatch_loss = scaled_loss * gradient_accumulation_steps 

    elif length_norm_type == "mask_normalize":
        # --- Constant 逻辑：除以固定常数 C ---
        # 这种模式下，你需要维持原有的步长缩放逻辑
        per_example_loss = (per_token_loss * response_mask).sum(dim=1) / constant_normalizer
        microbatch_loss = per_example_loss.mean()
        scaled_loss = microbatch_loss / gradient_accumulation_steps

    else: # mask_mean
        # --- 默认逻辑：句子内 Token 平均 ---
        per_example_loss = masked_mean(per_token_loss, response_mask, dim=1)
        microbatch_loss = per_example_loss.mean()
        scaled_loss = microbatch_loss / gradient_accumulation_steps
    
    # 3. 反向传播
    scaled_loss.backward()
    
    # 4. 准备元数据
    metadata = {
        "loss": microbatch_loss.detach(),
    }
    metadata.update(loss_metadata)
    
    return scaled_loss, metadata

def log_generations(
    vllm_model: LLM,
    sampling_params: SamplingParams,
    prompts: List[str],
    ground_truths: List[str],
    reward_fn: Callable[[str, str], Dict[str, float]],
    step: int,
    log_prefix: str = "eval"
):
    """
    让模型生成回答并记录详细的评估指标。
    """
    # 1. 模型生成回答
    outputs = vllm_model.generate(prompts, sampling_params)
    
    table_data = []
    all_lengths = []
    correct_lengths = []
    incorrect_lengths = []
    total_reward = 0
    total_format_reward = 0
    total_answer_reward = 0
    
    # 2. 逐条处理生成结果
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        gold_answer = ground_truths[i]
        
        # 计算奖励
        scores = reward_fn(generated_text, gold_answer)
        
        r  = scores.get("reward", 0.0)
        fr = scores.get("format_reward", 0.0)
        ar = scores.get("answer_reward", 0.0)
        
        resp_len = len(generated_text)
        all_lengths.append(resp_len)
        
        if r > 0.5:
            correct_lengths.append(resp_len)
        else:
            incorrect_lengths.append(resp_len)
            
        total_reward += r
        total_format_reward += fr
        total_answer_reward += ar

        # 展示前 10 条到 WandB Table
        if i < 10: 
            table_data.append([
                step, 
                prompts[i],
                generated_text, 
                gold_answer, 
                r, fr, ar
            ])

    # 3. 计算聚合统计量
    metrics = {
        f"{log_prefix}/accuracy": total_reward / len(prompts),
        f"{log_prefix}/format_score": total_format_reward / len(prompts),
        f"{log_prefix}/answer_score": total_answer_reward / len(prompts),
        f"{log_prefix}/avg_length": np.mean(all_lengths),
        f"{log_prefix}/avg_length_correct": np.mean(correct_lengths) if correct_lengths else 0,
        f"{log_prefix}/avg_length_incorrect": np.mean(incorrect_lengths) if incorrect_lengths else 0,
        f"{log_prefix}_step": step 
    }

    # 4. 记录到日志系统
    if wandb.run is not None:
        # 记录表格
        columns = ["grpo_step", "prompt", "response", "ground_truth", "reward", "format_reward", "answer_reward"]
        wandb.log({f"{log_prefix}/samples": wandb.Table(columns=columns, data=table_data)}, step=step)
        # 记录标量数值
        wandb.log(metrics, step=step)
    
    print(f"Step {step}: Accuracy: {metrics[f'{log_prefix}/accuracy']:.4f}, Avg Len: {metrics[f'{log_prefix}/avg_length']:.1f}")

    return metrics