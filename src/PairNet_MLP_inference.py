import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train')))

# ----------------- Model -----------------
class PairNet(nn.Module):
    def __init__(self, input_dim, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim  # 假设每个embedding的维度是1280
        self.hidden_dim = input_dim // 5  # 隐藏层维度
        self.last_dim = self.input_dim + self.hidden_dim  # 主embedding与位点embedding拼接后的维度
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
        sup1 = self.linear(sup1)  # 位点embedding
        sup2 = self.linear(sup2)
        x1 = torch.cat([x1, sup1], dim=-1)  # 将主embedding与位点embedding拼接 1536
        x2 = torch.cat([x2, sup2], dim=-1)  # 将主embedding与位点embedding拼接
        x = torch.cat([x1, x2], dim=-1)
        return self.net(x).squeeze(-1)


# ----------- 单个进程执行逻辑 -----------
def ddp_worker(rank: int, world_size: int,
               model_path: Path, input_dim: int,
               main_data_dir: Path, json_path: Path,
               threshold: float, output_dir: Path,
               batch_size: int, cache_path: Path):

    # 初始化 NCCL 通信组和当前设备
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 加载模型并去除 DDP 的 'module.' 前缀
    model = PairNet(1280).to(device)
    state_dict = torch.load(model_path, map_location="cpu")
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # 读取 embedding 缓存（提前构建好的）
    cache = torch.load(cache_path, map_location="cpu")
    variant_ids = sorted(cache.keys())
    id2index = {vid: idx for idx, vid in enumerate(variant_ids)}
    all_embeds = [(cache[vid][0].to(device), cache[vid][1].to(device)) for vid in variant_ids]
    print(f"[RANK {rank}] Loaded {len(all_embeds)} embeddings from cache.")

    # 构建所有正反 pair 对 [(i, j), (j, i)]
    pair_list = []
    print(f"[RANK {rank}] Generating pair list...")
    for i in range(len(variant_ids)):
        for j in range(len(variant_ids)):
            if i != j:
                pair_list.append((i, j))

    # 将所有 pair 均匀分配到 rank 上处理
    local_pair_list = pair_list[rank::world_size]

    print(f"[RANK {rank}] Start processing {len(local_pair_list)} pairs...")
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
                print(f"[RANK {rank}] Processing pair ({i}, {j})")
                score = scores[idx].item()
                vid_i = variant_ids[i]
                vid_j = variant_ids[j]

                if score > threshold:
                    win_counts[vid_i] += 1
                elif score < threshold:
                    win_counts[vid_j] += 1

                total_counts[vid_i] += 1
                total_counts[vid_j] += 1

    # 计算当前 rank 的胜率表并保存
    records = []
    for vid in sorted(win_counts):
        win_rate = win_counts[vid] / total_counts[vid]
        records.append((vid, win_rate))
    df = pd.DataFrame(records, columns=["variant_id", "win_rate"])
    df.to_csv(output_dir / f"rank{rank}_winrates.csv", index=False)

    print(f"[RANK {rank}] Done. Written to CSV.")
    dist.barrier()
    dist.destroy_process_group()


# ----------- 主启动逻辑 -----------
def main():
    
    import argparse
    parser = argparse.ArgumentParser(description="Distributed PairNet Inference")
    parser.add_argument("--main_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_weight", type=str, required=True)
    args = parser.parse_args()
    # ----------- 参数设置 -----------
    bs = 32
    mode = "mlp"
    num_gpus = 1
    input_dim = 1280
    batch_size = 4096  # 新增：batch 推理大小
    model_path = Path(args.model_weight)
    main_data_dir = Path(args.main_dir)
    json_path = Path(main_data_dir) / "data_point.json"
    output_dir = Path(args.output_dir) / f"mlp_inference_old_config"
    output_dir.mkdir(exist_ok=True)
    threshold = 0.0
    cache_path = Path(f"{args.main_dir}/embedding_cache.pt")

    # 如果 embedding 缓存不存在，则创建
    if not cache_path.exists():
        print("[MAIN] Creating embedding cache...")
        with open(json_path, "r") as f:
            metric_dict = {
                d['Variants'] if 'Variants' in d else d['name']: d['active_numbers']
                for d in json.load(f)
            }

        all_paths = sorted(list(main_data_dir.glob("*.npy")))
        cache = {}
        for p in all_paths:
            vid = p.stem
            arr = np.load(p)
            t = torch.tensor(arr, dtype=torch.float32)
            emb = t[1:-1].mean(dim=0) if t.ndim > 1 else t
            emb = F.normalize(emb, dim=-1)
            sup = torch.stack([t[pos] for pos in metric_dict[vid]], dim=0).mean(dim=0)
            sup = F.normalize(sup, dim=-1)
            cache[vid] = (emb, sup)
        torch.save(cache, cache_path)
        print("[MAIN] Embedding cache saved.")

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    # 启动多进程，每张 GPU 一进程
    mp.spawn(
        ddp_worker,
        args=(num_gpus, model_path, input_dim, main_data_dir, json_path, threshold, output_dir, batch_size, cache_path),
        nprocs=num_gpus,
        join=True
    )

    # 合并每个 rank 的 CSV 文件为总表
    print("[MAIN] Merging results...")
    import glob
    files = glob.glob(str(output_dir / "rank*_winrates.csv"))
    dfs = [pd.read_csv(f) for f in files]
    final_df = pd.concat(dfs).groupby("variant_id", as_index=False).mean()
    final_df = final_df.sort_values("win_rate", ascending=False)
    final_df.to_csv(output_dir / "final_sorted_winrates.csv", index=False)
    print(f"[MAIN] Done! Final result saved to {output_dir / 'final_sorted_winrates.csv'}")


if __name__ == "__main__":
    main()
