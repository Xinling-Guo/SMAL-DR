# task6_utils.py
import os
import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train')))

from PairNet_Transformer_train import PairNet  # 复用你已有的 PairNet


# ===============================================================
# 单机版 Inference，不改变你的原始推理逻辑
# ===============================================================
def inference(model_weight: Path,
              main_data_dir: Path,
              cache_path: Path,
              output_dir: Path,
              config: dict):
    """
    单机版 inference，保持你的逻辑不变：
    1. 加载模型权重
    2. 加载 embedding cache
    3. 构造所有 pair (i,j)
    4. 执行推理
    5. 保存 rank0_winrates.csv
    """

    # ------------------------------------
    # 读取参数
    # ------------------------------------
    threshold = config["task6_config"].get("threshold", 0.0)
    batch_size = config["task6_config"].get("batch_size", 4096)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TASK6] Using device: {device}")

    # ------------------------------------
    # 加载模型
    # ------------------------------------
    print(f"[TASK6] Loading model from {model_weight}")
    model = PairNet().to(device)

    state_dict = torch.load(model_weight, map_location="cpu")
    # 去掉 DDP 的 module.
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # ------------------------------------
    # 加载 embedding cache
    # ------------------------------------
    print(f"[TASK6] Loading embedding cache from {cache_path}")
    cache = torch.load(cache_path, map_location="cpu")

    # variant id 排序
    variant_ids = sorted(cache.keys())
    all_embeds = [(cache[vid][0].to(device), cache[vid][1].to(device)) for vid in variant_ids]
    print(f"[TASK6] Loaded {len(all_embeds)} embeddings.")

    # ------------------------------------
    # 构造所有 pair
    # ------------------------------------
    pair_list = [(i, j) for i in range(len(variant_ids)) for j in range(len(variant_ids)) if i != j]
    print(f"[TASK6] Total pairs: {len(pair_list)}")

    # ------------------------------------
    # 推理
    # ------------------------------------
    win_counts = defaultdict(int)
    total_counts = defaultdict(int)

    with torch.no_grad():
        for b in range(0, len(pair_list), batch_size):
            batch_pairs = pair_list[b:b + batch_size]

            g1_batch = torch.stack([all_embeds[i][0] for (i, j) in batch_pairs])
            g2_batch = torch.stack([all_embeds[j][0] for (i, j) in batch_pairs])
            s1_batch = torch.stack([all_embeds[i][1] for (i, j) in batch_pairs])
            s2_batch = torch.stack([all_embeds[j][1] for (i, j) in batch_pairs])

            scores = model(g1_batch, g2_batch, s1_batch, s2_batch).cpu()

            for k, (i, j) in enumerate(batch_pairs):
                score = scores[k].item()
                vid_i = variant_ids[i]
                vid_j = variant_ids[j]

                if score > threshold:
                    win_counts[vid_i] += 1
                elif score < threshold:
                    win_counts[vid_j] += 1

                total_counts[vid_i] += 1
                total_counts[vid_j] += 1

    # ------------------------------------
    # 保存结果（模拟 rank0）
    # ------------------------------------
    print(f"[TASK6] Saving results to rank0_winrates.csv")

    records = []
    for vid in sorted(win_counts):
        win_rate = win_counts[vid] / total_counts[vid]
        records.append((vid, win_rate))

    df = pd.DataFrame(records, columns=["variant_id", "win_rate"])
    df.to_csv(output_dir / "rank0_winrates.csv", index=False)

    print("[TASK6] Inference finished successfully!")
