import os
import random
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.stats import kendalltau, spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------- Utilities -----------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[SEED] Global seed set to {seed}")

# ----------------- Dataset -----------------
class CustomPairwiseDataset(Dataset):
    """
    单机版 Dataset，不含分布式逻辑。
    """
    def __init__(self, data_dir: Path, metric_name, diff_threshold=0.0):
        emb_dir = data_dir / "Esm2Embedding-sp-1280"
        assert emb_dir.exists(), f"Embedding directory not found: {emb_dir}"

        self.pdb_ids = sorted([f.stem for f in emb_dir.glob("*.npy")])
        print(f"[DATA] Found {len(self.pdb_ids)} samples")

        # Load metric json
        metric_df = pd.read_json(emb_dir / "data_point.json")

        self.embeddings = []
        self.embeddings_sup = []
        self.values = []

        for pdb_id in tqdm(self.pdb_ids, desc="Loading samples"):
            # 获取指标
            val = np.mean([
                metric_df.loc[metric_df['Variants'] == pdb_id, m].values[0]
                for m in metric_name
            ])
            self.values.append(val)

            # 加载 embedding
            emb = np.load(emb_dir / f"{pdb_id}.npy")
            emb = torch.tensor(emb, dtype=torch.float32)

            # --- sup embedding 3 位点 ---
            order = metric_df.loc[metric_df['Variants'] == pdb_id, 'active_numbers'].values[0]
            emb_sup = torch.stack([emb[order[0]], emb[order[1]], emb[order[2]]], dim=0).mean(0)
            emb_sup = F.normalize(emb_sup, p=2, dim=-1)
            self.embeddings_sup.append(emb_sup)

            # --- main embedding 平均 ---
            emb_main = emb[1:-1] if emb.ndim > 1 else emb
            emb_main = emb_main.mean(dim=0)
            emb_main = F.normalize(emb_main, p=2, dim=-1)
            self.embeddings.append(emb_main)

        # 生成 pair 数据
        self.pairs = self._generate_pairs(self.values, diff_threshold)
        print(f"[DATA] Generated {len(self.pairs)} training pairs")

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
        emb1 = self.embeddings[i]
        emb2 = self.embeddings[j]
        sup1 = self.embeddings_sup[i]
        sup2 = self.embeddings_sup[j]
        label = 1 if self.values[i] > self.values[j] else -1
        return emb1, emb2, sup1, sup2, torch.tensor(label, dtype=torch.float32)

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

    def forward(self, z, sup):
        sup_proj = F.relu(self.sup_proj(sup))  # [B,256]
        tokens = torch.stack([z, sup_proj], dim=1)  # [B,2,256]
        fused = self.transformer(tokens)  # [B,2,256]
        pooled = fused.mean(dim=1)  # [B,256]
        return pooled

# ----------------- Pair Model -----------------
class PairNet(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(1280, 256)
        self.fusion = FusionTransformer(emb_dim=256)
        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, g1, g2, sup1, sup2):
        z1 = self.linear(g1)
        z2 = self.linear(g2)
        f1 = self.fusion(z1, sup1)
        f2 = self.fusion(z2, sup2)
        x = torch.cat([f1, f2], dim=-1)
        return self.net(x).squeeze(-1)

# ----------------- Training -----------------
def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for g1, g2, sup1, sup2, y in tqdm(loader, desc="Training"):
        # Move to device
        g1, g2 = g1.to(device), g2.to(device)
        sup1, sup2 = sup1.to(device), sup2.to(device)
        y = y.to(device)

        scores = model(g1, g2, sup1, sup2)
        loss = loss_fn(scores, torch.zeros_like(scores), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)

    return total_loss / len(loader.dataset)

# ----------------- Evaluation -----------------
def eval_metrics(model, loader, loss_fn, device):
    model.eval()
    true_labels = []
    pred_diff = []
    correct = 0
    total_loss = 0

    with torch.no_grad():
        for g1, g2, sup1, sup2, y in tqdm(loader, desc="Evaluating"):
            g1, g2 = g1.to(device), g2.to(device)
            sup1, sup2 = sup1.to(device), sup2.to(device)
            y = y.to(device)

            scores = model(g1, g2, sup1, sup2)
            loss = loss_fn(scores, torch.zeros_like(scores), y)

            total_loss += loss.item() * y.size(0)
            preds = torch.sign(scores).cpu().numpy()
            gt = y.cpu().numpy()

            correct += (preds == gt).sum()
            true_labels.extend(gt.tolist())
            pred_diff.extend(scores.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    tau = kendalltau(true_labels, pred_diff).correlation
    rho = spearmanr(true_labels, pred_diff).correlation

    print(f"[Eval] Loss={avg_loss:.4f}, Acc={acc:.4f}, Tau={tau:.4f}, Rho={rho:.4f}")
    return avg_loss, acc, tau, rho
