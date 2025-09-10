import os
import random
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.stats import kendalltau, spearmanr
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from torch_geometric.nn import GCNConv, global_mean_pool
# from torch_geometric.data import Data as GraphData, Batch
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ----------------- Utilities -----------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[SEED] Global seed set to {seed}")

# ----------------- Dataset -----------------
class CustomPairwiseDataset(Dataset):
    def __init__(self, pdb_ids, labels, data_dir: Path, structure_dir: Path, metric_name, diff_threshold=0.0, rank=0, world_size=1):
        self.pdb_ids = pdb_ids
        self.labels = labels
        self.embeddings = []
        self.graphs = []
        self.values = []
        self.embeddings_sup = []
        
        print(f"[DATA] Loading dataset with {len(pdb_ids)} samples on rank {rank}")
        
        # 确保所有进程同步开始加载数据
        dist.barrier()
        
        if rank == 0:
            # 主进程加载metric_df
            metric_df = pd.read_json(data_dir / "data_point.json")
            # 将DataFrame转换为JSON字符串
            json_str = metric_df.to_json().encode()
            # 创建一个足够大的缓冲区tensor
            max_size = len(json_str)
            # 广播最大尺寸
            size_tensor = torch.tensor([max_size], dtype=torch.long, device=torch.device(f"cuda:{rank}"))
            dist.broadcast(size_tensor, src=0)
            # 创建并填充数据tensor
            data_tensor = torch.zeros(max_size, dtype=torch.uint8, device=torch.device(f"cuda:{rank}"))
            data_tensor[:max_size] = torch.tensor(list(json_str), dtype=torch.uint8, device=torch.device(f"cuda:{rank}"))
            # 广播数据tensor
            dist.broadcast(data_tensor, src=0)
        else:
            # 从主进程接收数据尺寸
            size_tensor = torch.tensor([0], dtype=torch.long, device=torch.device(f"cuda:{rank}"))
            dist.broadcast(size_tensor, src=0)
            max_size = size_tensor.item()
            # 创建接收缓冲区
            data_tensor = torch.zeros(max_size, dtype=torch.uint8, device=torch.device(f"cuda:{rank}"))
            # 接收数据
            dist.broadcast(data_tensor, src=0)
            # 转换为JSON字符串
            json_str = data_tensor.cpu().numpy().tobytes().decode()
            # 解析为DataFrame
            metric_df = pd.read_json(json_str)
        
        # 同步所有进程，确保所有进程都已成功获取metric_df
        dist.barrier()
        
        # 为每个进程添加进度条
        total = len(pdb_ids)
        step = total // 10
        if step == 0:
            step = 1
            
        for i, pdb_id in enumerate(pdb_ids):
            if i % step == 0 and rank == 0:
                print(f"[DATA] Loading sample {i}/{total}")
                
            val = np.mean([metric_df.loc[metric_df['Variants'] == pdb_id, m].values[0] for m in metric_name])
            self.values.append(val)

            emb = np.load(data_dir / "Esm2Embedding-sp-1280" / f"{pdb_id}.npy")
            emb = torch.tensor(emb, dtype=torch.float32)
            
            order = metric_df.loc[metric_df['Variants'] == pdb_id, 'active_numbers'].values[0]
            emb_sup = torch.stack([emb[order[0]], emb[order[1]], emb[order[2]]], dim=0).mean(0)
            emb_sup = F.normalize(emb_sup, p=2, dim=-1)
            emb = emb[1:-1] if emb.ndim > 1 else emb
            emb = emb.mean(dim=0)
            emb = F.normalize(emb, p=2, dim=-1)
            self.embeddings.append(emb)

            self.embeddings_sup.append(emb_sup)

        self.pairs = self._generate_pairs(self.values, diff_threshold)
        print(f"[DATA] Generated {len(self.pairs)} pairs on rank {rank}")

    def _generate_pairs(self, values, diff_threshold):
        pairs = []
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                if abs(values[i] - values[j]) >= diff_threshold:
                    pairs.append((i, j))
                    pairs.append((j, i))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        # g1 = self.graphs[i]
        # g2 = self.graphs[j]
        emb1 = self.embeddings[i]
        emb2 = self.embeddings[j]
        emb1_sup = self.embeddings_sup[i]
        emb2_sup = self.embeddings_sup[j]
        label = 1 if self.values[i] > self.values[j] else -1
        return emb1, emb2, emb1_sup, emb2_sup, torch.tensor(label, dtype=torch.float32)

# ----------------- Fusion Transformer -----------------
class FusionTransformer(nn.Module):
    def __init__(self, emb_dim=256, num_heads=4, ff_dim=512):
        super().__init__()
        self.sup_proj = nn.Linear(1280, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        print(f"[MODEL] Initialized FusionTransformer with {sum(p.numel() for p in self.parameters())} parameters")

    def forward(self, z, sup):
        sup_proj = F.relu(self.sup_proj(sup))  # [B,256]
        tokens = torch.stack([z, sup_proj], dim=1)  # [B,2,256]
        fused = self.transformer(tokens)  # [B,2,256]
        pooled = fused.mean(dim=1)  # [B,256]
        return pooled

# ----------------- Pair Model -----------------
class PairNet(nn.Module):
    def __init__(self, gcn_hidden=256, emb_sup_dim=1280, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(1280, 256)
        self.fusion = FusionTransformer(emb_dim=gcn_hidden)
        self.net = nn.Sequential(
            nn.Linear(2 * gcn_hidden, gcn_hidden),
            nn.BatchNorm1d(gcn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gcn_hidden, gcn_hidden // 2),
            nn.BatchNorm1d(gcn_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gcn_hidden // 2, 1)
        )
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[MODEL] Initialized PairNet with {total_params:,} total parameters")

    def forward(self, g1, g2, sup1, sup2):
        z1 = self.linear(g1)
        z2 = self.linear(g2)
        f1 = self.fusion(z1, sup1)
        f2 = self.fusion(z2, sup2)
        x = torch.cat([f1, f2], dim=-1)
        return self.net(x).squeeze(-1)

# ----------------- Training -----------------
def train_epoch(model, loader, optimizer, loss_fn, device, rank, epoch, total_epochs):
    model.train()
    total_loss = 0
    batch_count = len(loader)
    
    # 创建tqdm进度条，只在主进程显示
    if rank == 0:
        progress_bar = tqdm(enumerate(loader), total=batch_count, 
                           desc=f"[TRAIN] Epoch {epoch}/{total_epochs}", 
                           position=0, leave=True)
    else:
        progress_bar = enumerate(loader)
    
    for i, (g1, g2, sup1, sup2, y) in progress_bar:
        g1, g2 = g1.to(device), g2.to(device)
        sup1, sup2, y = sup1.to(device), sup2.to(device), y.to(device)
        
        scores = model(g1, g2, sup1, sup2)
        loss = loss_fn(scores, torch.zeros_like(scores), y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * y.size(0)
        
        # 更新进度条信息
        if rank == 0:
            avg_loss = total_loss / ((i + 1) * y.size(0))
            progress_bar.set_postfix({"Loss": f"{avg_loss:.4f}"})
    
    # 关闭进度条
    if rank == 0:
        progress_bar.close()
    
    return total_loss / len(loader.dataset)

# ----------------- Evaluation -----------------
def eval_metrics(model, loader, loss_fn, device, rank, phase="Validation"):
    model.eval()
    true_labels = []
    pred_diff = []
    correct = 0
    total_loss = 0
    batch_count = len(loader)
    
    # 创建tqdm进度条，只在主进程显示
    if rank == 0:
        progress_bar = tqdm(enumerate(loader), total=batch_count, 
                           desc=f"[{phase}]", 
                           position=0, leave=True)
    else:
        progress_bar = enumerate(loader)
    
    with torch.no_grad():
        for i, (g1, g2, sup1, sup2, y) in progress_bar:
            g1, g2 = g1.to(device), g2.to(device)
            sup1, sup2, y = sup1.to(device), sup2.to(device), y.to(device)
            
            scores = model(g1, g2, sup1, sup2)
            loss = loss_fn(scores, torch.zeros_like(scores), y)
            
            total_loss += loss.item() * y.size(0)
            preds = torch.sign(scores).cpu().numpy()
            gt = y.cpu().numpy()
            correct += (preds == gt).sum()
            true_labels.extend(gt.tolist())
            pred_diff.extend(scores.cpu().tolist())
            
            # 更新进度条信息
            if rank == 0:
                avg_loss = total_loss / ((i + 1) * y.size(0))
                progress_bar.set_postfix({"Loss": f"{avg_loss:.4f}"})
    
    # 关闭进度条
    if rank == 0:
        progress_bar.close()
    
    avg_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    tau = kendalltau(true_labels, pred_diff).correlation
    rho = spearmanr(true_labels, pred_diff).correlation
    
    if rank == 0:
        print(f"[{phase}] Results - Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}, "
              f"Kendall's Tau: {tau:.4f}, Spearman's Rho: {rho:.4f}")
    
    return avg_loss, acc, tau, rho

# ----------------- Main -----------------
def main(rank, world_size, seed, config):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler
    from pathlib import Path
    import pandas as pd

    print(f"[RANK {rank}] Initializing distributed training with {world_size} GPUs")
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    set_seed(seed)

    DATA_DIR = Path("/root/autodl-tmp/PairNet/data")
    STRUCTURE_DIR = Path("/root/autodl-tmp/PairNet/data/GCN_0N1/sp")
    RESULTS_DIR = Path("/root/autodl-tmp/PairNet/model_weights/train_seq_no_struc_transformer_all_wet_data_all_seq_3_site")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    RESULTS_DIR = RESULTS_DIR / f"1gpu_SEED_42_BATCHSIZE_32"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metric = ["FITNESS"]
    POS_DIR = Path("/root/autodl-tmp/PairNet/data/sp_pdb_kfold/pos")
    NEG_DIR = Path("/root/autodl-tmp/PairNet/data/sp_pdb_kfold/neg")

    pos_ids = [f.stem for f in POS_DIR.glob("*.npy")]
    neg_ids = [f.stem for f in NEG_DIR.glob("*.npy")]

    train_ids = pos_ids + neg_ids
    train_labels = [1]*len(pos_ids) + [0]*len(neg_ids)

    dist.barrier()
    print(f"[RANK {rank}] Creating full training dataset")
    train_ds = CustomPairwiseDataset(train_ids, train_labels, DATA_DIR, STRUCTURE_DIR, metric, config["diff_threshold"], rank, world_size)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], sampler=train_sampler, num_workers=4, pin_memory=True, persistent_workers=True)

    model = PairNet().to(device)
    model = DDP(model, device_ids=[rank])

    loss_fn = torch.nn.MarginRankingLoss(margin=config["margin"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_train_loss = float('inf')
    no_improve = 0

    for epoch in range(1, config["epochs"] + 1):
        train_sampler.set_epoch(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        if rank == 0:
            print(f"\n[RANK {rank}] Epoch {epoch}/{config['epochs']} - LR: {current_lr:.6f}")

        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, rank, epoch, config["epochs"])
        dist.barrier()

        early_stop = torch.tensor(0, device=device)

        if rank == 0:
            scheduler.step(train_loss)
            if train_loss < best_train_loss:
                print(f"[RANK {rank}] New best training loss: {train_loss:.4f} (prev: {best_train_loss:.4f})")
                best_train_loss = train_loss
                no_improve = 0
                torch.save(model.state_dict(), RESULTS_DIR / "best_model.pth")
            else:
                no_improve += 1
                print(f"[RANK {rank}] No improvement for {no_improve}/{config['patience']} epochs")
                if no_improve >= config["patience"]:
                    print(f"[RANK {rank}] Early stopping triggered.")
                    early_stop.fill_(1)  # ✅ 正确方式

        # 广播 early_stop 信号给所有进程
        dist.broadcast(early_stop, src=0)

        if early_stop.item() == 1:
            break

    dist.barrier()

    # === 排序推理：所有进程一起执行 ===
    model.load_state_dict(torch.load(RESULTS_DIR / "best_model.pth", map_location=device))
    model.eval()

    train_emb = train_ds.embeddings
    train_sup = train_ds.embeddings_sup
    train_pdb_ids = train_ds.pdb_ids
    train_values = train_ds.values
    n_train = len(train_emb)

    wins1 = [0 for _ in range(n_train)]
    wins2 = [0 for _ in range(n_train)]
    start_idx = (rank * n_train) // world_size
    end_idx = ((rank + 1) * n_train) // world_size

    with torch.no_grad():
        for i in range(start_idx, end_idx):
            for j in range(n_train):
                if i == j: continue
                g1 = train_emb[i].unsqueeze(0).to(device)
                g2 = train_emb[j].unsqueeze(0).to(device)
                sup1 = train_sup[i].unsqueeze(0).to(device)
                sup2 = train_sup[j].unsqueeze(0).to(device)
                score = model(g1, g2, sup1, sup2).item()
                if score > 0: wins1[i] += 1
                else: wins1[j] += 1

        for i in range(start_idx, end_idx):
            for j in range(n_train):
                if i == j: continue
                g1 = train_emb[j].unsqueeze(0).to(device)
                g2 = train_emb[i].unsqueeze(0).to(device)
                sup1 = train_sup[j].unsqueeze(0).to(device)
                sup2 = train_sup[i].unsqueeze(0).to(device)
                score = model(g1, g2, sup1, sup2).item()
                if score > 0: wins2[j] += 1
                else: wins2[i] += 1

    dist.barrier()
    all_wins1 = [torch.zeros(n_train, dtype=torch.long, device=device) for _ in range(world_size)]
    all_wins2 = [torch.zeros(n_train, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_wins1, torch.tensor(wins1, dtype=torch.long, device=device))
    dist.all_gather(all_wins2, torch.tensor(wins2, dtype=torch.long, device=device))

    wins1 = [0] * n_train
    wins2 = [0] * n_train
    for w in all_wins1:
        wins1 = [x + y for x, y in zip(wins1, w.cpu().tolist())]
    for w in all_wins2:
        wins2 = [x + y for x, y in zip(wins2, w.cpu().tolist())]

    win_rates1 = [w / (n_train - 1) for w in wins1]
    win_rates2 = [w / (n_train - 1) for w in wins2]

    sorted_idx1 = sorted(range(n_train), key=lambda k: win_rates1[k], reverse=True)
    sorted_idx2 = sorted(range(n_train), key=lambda k: win_rates2[k], reverse=True)
    rank1 = [0] * n_train
    rank2 = [0] * n_train
    for pos, idx in enumerate(sorted_idx1): rank1[idx] = pos
    for pos, idx in enumerate(sorted_idx2): rank2[idx] = pos
    avg_rank = [(rank1[i] + rank2[i]) / 2.0 for i in range(n_train)]
    sorted_idx_avg = sorted(range(n_train), key=lambda k: avg_rank[k])

    if rank == 0:
        df = pd.DataFrame({
            'pdb_id': [train_pdb_ids[i] for i in sorted_idx_avg],
            'true_value': [train_values[i] for i in sorted_idx_avg],
            'avg_rank': [avg_rank[i] for i in sorted_idx_avg],
            'win_rate1': [win_rates1[i] for i in sorted_idx_avg],
            'win_rate2': [win_rates2[i] for i in sorted_idx_avg],
            'rank1': [rank1[i] for i in sorted_idx_avg],
            'rank2': [rank2[i] for i in sorted_idx_avg]
        })
        csv_path = RESULTS_DIR / "ranking_train.csv"
        df.to_csv(csv_path, index=False)
        print(f"[RANK {rank}] Saved full training ranking to {csv_path}")

    dist.barrier()
    print(f"[RANK {rank}] Training completed. Best train loss: {best_train_loss:.4f}")
    dist.destroy_process_group()




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Distributed Training for PairNet')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--config', type=str, default='650m', help='Config name')
    parser.add_argument('--size', type=int, default=32, help='Batch size for training')
    args = parser.parse_args()
    
    print(f"Starting PairNet training with seed {args.seed} and config {args.config}")
    
    # 获取GPU数量
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise ValueError("No GPU available for distributed training")
    
    print(f"Found {world_size} CUDA devices")
    
    # 获取torchrun设置的环境变量
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', world_size))
    
    print(f"Running in distributed mode. Rank {rank}/{world_size}, Local Rank {local_rank}")
    main(rank, world_size, args.seed, {
        "batch_size": args.size,
        "lr": 3e-5,
        "weight_decay": 1e-4,
        "margin": 0.5,
        "diff_threshold": 0.0,
        "hidden_dim": 256,
        "epochs": 100,
        "patience": 20,
        # "folds": 5
    })