from ast import List
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import kendalltau, spearmanr
# import wandb

# ----------------- Utilities -----------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------- Dataset -----------------
class PairwiseDataset(Dataset):
    def __init__(self, dataset_dir: Path, metric_name: List, diff_threshold: float = 0.0, embeddings_dir: Path = None):
        self.pdb_ids, self.embeddings, self.embeddings_sup, self.labels = self._load_embeddings(dataset_dir, metric_name)
        self.pairs = self._generate_pairs(self.labels, diff_threshold)

    def _load_embeddings(self, data_dir: Path, metric_name: List):
        files = sorted(embeddings_dir.glob("*.npy"))
        metric_file = embeddings_dir / "data_point.json"
        metric_df = pd.read_json(metric_file)

        pdb_ids = []
        embeddings = []
        embeddings_sup = []
        labels = []
        for pdb_file in tqdm(files, desc=f"Loading"):
            pdb_id = pdb_file.stem
            pdb_ids.append(pdb_id)
            
            val = np.mean([metric_df.loc[metric_df['Variants'] == pdb_id, m].values[0] for m in metric_name])
            
            labels.append(val)
            emb = torch.tensor(np.load(embeddings_dir / f"{pdb_id}.npy"), dtype=torch.float32)

            # 获取三个位点的embeddings
            order = metric_df.loc[metric_df['Variants'] == pdb_id, 'active_numbers'].values[0]
            emb_sup = torch.stack([emb[order[0]], emb[order[1]], emb[order[2]]], dim=0)  # 三个位置的embedding
            emb_sup = emb_sup.mean(dim=0)  # 平均
            emb_sup = F.normalize(emb_sup, p=2, dim=-1)
            embeddings_sup.append(emb_sup)  # 3*1280 -> 1280 -> 256+1280
            
            emb = emb[1:-1]  # 去掉第一个和最后一个位置的embedding
            emb = emb.mean(dim=0)
            emb = F.normalize(emb, p=2, dim=-1)
            embeddings.append(emb)  # （1280， ）

        return pdb_ids, embeddings, embeddings_sup, labels

    def _generate_pairs(self, labels, diff_threshold: float):
        pairs = []
        n = len(labels)
        for i in range(n):
            for j in range(i+1, n):
                if abs(labels[i] - labels[j]) >= diff_threshold:
                    pairs.append((i, j))
                    pairs.append((j, i))  # Add reverse pair for symmetry
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        emb1, emb2 = self.embeddings[i], self.embeddings[j]
        emb1_sup = self.embeddings_sup[i]
        emb2_sup = self.embeddings_sup[j]
        label = 1 if self.labels[i] > self.labels[j] else -1
        return emb1, emb2, emb1_sup, emb2_sup, torch.tensor(label, dtype=torch.float32)

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

# ----------------- Training -----------------
def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for x1, x2, sup1, sup2, y in loader:
        x1, x2, sup1, sup2, y = x1.to(device), x2.to(device), sup1.to(device), sup2.to(device), y.to(device)
        scores = model(x1, x2, sup1, sup2)
        loss = loss_fn(scores, torch.zeros_like(scores), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x1.size(0)
    return total_loss / len(loader.dataset)

# ----------------- Evaluation -----------------
def eval_metrics(model, loader, loss_fn, device):
    model.eval()
    true_labels = []
    pred_diff = []
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for x1, x2, sup1, sup2, y in loader:
            x1, x2, sup1, sup2, y = x1.to(device), x2.to(device), sup1.to(device), sup2.to(device), y.to(device)
            scores = model(x1, x2, sup1, sup2)
            loss = loss_fn(scores, torch.zeros_like(scores), y)
            total_loss += loss.item() * x1.size(0)
            preds = torch.sign(scores).cpu().numpy()
            gt = y.cpu().numpy()
            correct += (preds == gt).sum()
            true_labels.extend(gt.tolist())
            pred_diff.extend(scores.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    tau = kendalltau(true_labels, pred_diff).correlation
    rho = spearmanr(true_labels, pred_diff).correlation
    return avg_loss, acc, tau, rho

# ----------------- Main -----------------
if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = Path("/root/autodl-tmp/PairNet/data")
    # RESULTS_DIR = Path(f"/root/autodl-tmp/PairNet/model_weights/all_wet_data_train_mlp_seq_no_struc/BATCHSIZE_32")
    RESULTS_DIR = Path(f"/root/autodl-tmp/PairNet/model_weights/train_seq_no_struc_mlp_all_wet_data_all_seq_3_site")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR = RESULTS_DIR / f"1gpu_SEED_42_BATCHSIZE_32"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    metric = ["FITNESS"]  # 选择一个指标或者多个指标的均值
    embeddings_dir = DATA_DIR / f"Esm2Embedding-sp-1280"

    metric_name = "_".join(metric)

    CONFIG = {"batch_size":32, "lr":3e-5, "weight_decay":1e-4,
            "margin":0.5, "diff_threshold":0.0, "hidden_dim":256,
            "epochs":100, "patience":20}

    train_ds = PairwiseDataset(DATA_DIR,  metric, CONFIG["diff_threshold"], embeddings_dir)
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)

    model = PairNet(input_dim=train_ds.embeddings[0].shape[0]).to(device)
    loss_fn = nn.MarginRankingLoss(margin=CONFIG["margin"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    no_improve = 0

    best_model_path = RESULTS_DIR / f"best_model.pth"

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        scheduler.step(train_loss)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}")

        # 保存当前 epoch 模型
        # torch.save(model.state_dict(), RESULTS_DIR / f"model_epoch_{epoch}_seed_42_sp_{metric_name}.pth")

        # 更新最佳模型
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improve += 1
            if no_improve >= CONFIG["patience"]:
                print("Early stopping on validation loss.")
                break

    print("Training complete. Best model saved at:", best_model_path)
    # 保存最终模型
    # 使用最佳模型对数据进行排序

    # === 排序推理：所有进程一起执行 ===
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    train_emb = train_ds.embeddings
    train_sup = train_ds.embeddings_sup
    train_pdb_ids = train_ds.pdb_ids
    train_values = train_ds.labels
    n_train = len(train_emb)

    wins1 = [0 for _ in range(n_train)]
    wins2 = [0 for _ in range(n_train)]

    with torch.no_grad():
        for i in range(n_train):
            for j in range(n_train):
                if i == j: continue
                g1 = train_emb[i].unsqueeze(0).to(device)
                g2 = train_emb[j].unsqueeze(0).to(device)
                sup1 = train_sup[i].unsqueeze(0).to(device)
                sup2 = train_sup[j].unsqueeze(0).to(device)
                score = model(g1, g2, sup1, sup2).item()
                if score > 0: wins1[i] += 1
                else: wins1[j] += 1

        for i in range(n_train):
            for j in range(n_train):
                if i == j: continue
                g1 = train_emb[j].unsqueeze(0).to(device)
                g2 = train_emb[i].unsqueeze(0).to(device)
                sup1 = train_sup[j].unsqueeze(0).to(device)
                sup2 = train_sup[i].unsqueeze(0).to(device)
                score = model(g1, g2, sup1, sup2).item()
                if score > 0: wins2[j] += 1
                else: wins2[i] += 1

    # 分别计算 win rate 排序
    win_rates1 = [w / (n_train - 1) for w in wins1]
    win_rates2 = [w / (n_train - 1) for w in wins2]

    sorted_idx1 = sorted(range(n_train), key=lambda k: win_rates1[k], reverse=True)
    sorted_idx2 = sorted(range(n_train), key=lambda k: win_rates2[k], reverse=True)

    # 计算综合排序（index 相加后取平均）
    avg_ranks = [(sorted_idx1.index(i) + 1 + sorted_idx2.index(i) + 1) / 2 for i in range(n_train)]
    final_sorted_indices = sorted(range(n_train), key=lambda k: avg_ranks[k])


    df = pd.DataFrame({
        'pdb_id': [train_pdb_ids[i] for i in final_sorted_indices],
        'true_value': [train_values[i] for i in final_sorted_indices],
        'avg_rank': [avg_ranks[i] for i in final_sorted_indices],
        'win_rate1': [win_rates1[i] for i in final_sorted_indices],
        'win_rate2': [win_rates2[i] for i in final_sorted_indices],
        "rank1": [sorted_idx1.index(i) + 1 for i in final_sorted_indices],
        "rank2": [sorted_idx2.index(i) + 1 for i in final_sorted_indices]
    })
    csv_path = RESULTS_DIR / "ranking_train.csv"
    df.to_csv(csv_path, index=False)
    print(f"Successfully completed training and saved rankings to {csv_path}")