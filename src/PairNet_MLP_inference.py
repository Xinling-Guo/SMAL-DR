import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist

# ----------------- Model -----------------
class PairNet(nn.Module):
    def __init__(self, input_dim, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim // 5
        self.last_dim = self.input_dim + self.hidden_dim
        self.linear = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )
        self.net = nn.Sequential(
            nn.Linear(2*self.last_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.BatchNorm1d(self.hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim//2, 1)
        )

    def forward(self, x1, x2, sup1, sup2):
        sup1 = self.linear(sup1)
        sup2 = self.linear(sup2)
        x1 = torch.cat([x1, sup1], dim=-1)
        x2 = torch.cat([x2, sup2], dim=-1)
        x = torch.cat([x1, x2], dim=-1)
        return self.net(x).squeeze(-1)


# ----------- 单个进程执行逻辑 -----------
def ddp_worker(rank: int, world_size: int,
               model_path: Path, input_dim: int,
               main_data_dir: Path, json_path: Path,
               threshold: float, output_dir: Path,
               batch_size: int, cache_path: Path):

    # 初始化 NCCL
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 加载模型
    model = PairNet(input_dim).to(device)
    state_dict = torch.load(model_path, map_location="cpu")
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # 加载缓存
    cache = torch.load(cache_path, map_location="cpu")
    variant_ids = sorted(cache.keys())
    all_embeds = [(cache[vid][0].to(device), cache[vid][1].to(device)) for vid in variant_ids]

    # 构建所有 pair
    pair_list = [(i, j) for i in range(len(variant_ids)) for j in range(len(variant_ids)) if i != j]
    local_pair_list = pair_list[rank::world_size]

    win_counts = defaultdict(int)
    total_counts = defaultdict(int)

    with torch.no_grad():
        for b in range(0, len(local_pair_list), batch_size):
            batch_pairs = local_pair_list[b:b + batch_size]

            g1_batch = torch.stack([all_embeds[i][0] for (i, j) in batch_pairs])
            g2_batch = torch.stack([all_embeds[j][0] for (i, j) in batch_pairs])
            s1_batch = torch.stack([all_embeds[i][1] for (i, j) in batch_pairs])
            s2_batch = torch.stack([all_embeds[j][1] for (i, j) in batch_pairs])

            scores = model(g1_batch, g2_batch, s1_batch, s2_batch).cpu()

            for idx, (i, j) in enumerate(batch_pairs):
                score = scores[idx].item()
                vid_i = variant_ids[i]
                vid_j = variant_ids[j]

                if score > threshold:
                    win_counts[vid_i] += 1
                elif score < threshold:
                    win_counts[vid_j] += 1

                total_counts[vid_i] += 1
                total_counts[vid_j] += 1

    # 每个 rank 写自己的 CSV
    records = [(vid, win_counts[vid] / total_counts[vid]) for vid in win_counts]
    df = pd.DataFrame(records, columns=["variant_id", "win_rate"])
    df.to_csv(output_dir / f"rank{rank}_winrates.csv", index=False)

    dist.barrier()
    dist.destroy_process_group()


# ----------- 新的 inference API（供 SMALDRTask4 调用）-----------
def inference(model_weight, main_dir, cache_path, output_dir, config):

    model_weight = Path(model_weight)
    main_dir = Path(main_dir)
    cache_path = Path(cache_path)
    output_dir = Path(output_dir)

    # 从 config 中读取参数
    json_path = Path(config['task4_config']['json_path'])
    batch_size = config['task4_config'].get('batch_size', 4096)
    threshold = config['task4_config'].get('threshold', 0.0)
    num_gpus = config['task4_config'].get('num_gpus', 1)
    input_dim = config['task4_config'].get('input_dim', 1280)

    # 必须设置 NCCL 环境
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = config['task4_config'].get("master_port", "29500")

    # 直接启动多 GPU 推理
    mp.spawn(
        ddp_worker,
        args=(num_gpus,
              model_weight,
              input_dim,
              main_dir,
              json_path,
              threshold,
              output_dir,
              batch_size,
              cache_path),
        nprocs=num_gpus,
        join=True
    )

    print("[Inference] All ranks completed.")
