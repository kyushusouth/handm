from collections import defaultdict

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import coo_matrix
from torch.utils.data import DataLoader, Dataset

# PyTorch Geometric (PyG) のインポート
from torch_geometric.utils import to_scipy_sparse_matrix
from tqdm import tqdm

from .base import BaseModel


# 1. PyTorchでLightGCNモデルの構造を定義
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, n_layers=3):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        # ユーザーとアイテムの埋め込みを初期化
        self.embedding_user = nn.Embedding(num_users, embedding_dim)
        self.embedding_item = nn.Embedding(num_items, embedding_dim)
        # Xavier Glorotの方法で重みを初期化
        nn.init.xavier_uniform_(self.embedding_user.weight)
        nn.init.xavier_uniform_(self.embedding_item.weight)

    def forward(self, adj_matrix):
        # 複数レイヤーの埋め込みを保持するリスト
        all_layer_embeddings = []
        current_embeddings = torch.cat(
            [self.embedding_user.weight, self.embedding_item.weight], dim=0
        )
        all_layer_embeddings.append(current_embeddings)

        # グラフ畳み込み（情報伝播）をレイヤー数だけ繰り返す
        for layer in range(self.n_layers):
            current_embeddings = torch.sparse.mm(adj_matrix, current_embeddings)
            all_layer_embeddings.append(current_embeddings)

        # 全てのレイヤーの埋め込みを平均して最終的な埋め込みとする
        final_embeddings = torch.mean(torch.stack(all_layer_embeddings, dim=0), dim=0)

        # ユーザーとアイテムの最終的な埋め込みに分割
        final_user_embeddings, final_item_embeddings = torch.split(
            final_embeddings, [self.num_users, self.num_items]
        )
        return final_user_embeddings, final_item_embeddings


# 2. BPR LossのためのカスタムDataset
class BPRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, num_items: int):
        self.users = torch.LongTensor(df["user_idx"].values)
        self.pos_items = torch.LongTensor(df["item_idx"].values)
        self.num_items = num_items

        # 各ユーザーがインタラクションしたアイテムのセットを作成
        self.user_item_sets = df.groupby("user_idx")["item_idx"].apply(set)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.pos_items[idx]

        # ネガティブサンプリング: ユーザーが触っていないアイテムをランダムに選択
        while True:
            neg_item = np.random.randint(0, self.num_items)
            if neg_item not in self.user_item_sets[user.item()]:
                break

        return user, pos_item, torch.LongTensor([neg_item]).squeeze()


# 3. 既存のフレームワークに統合するためのラッパークラス
class LightGCNModel(BaseModel):
    def __init__(
        self,
        cfg,
        embedding_dim=64,
        n_layers=3,
        epochs=20,
        batch_size=2048,
        lr=0.001,
        reg=0.001,
    ):
        self.cfg = cfg
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.reg = reg  # BPR Lossのための正則化項の係数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.user_map = {}
        self.item_map = {}
        self.inverse_item_map = {}
        self.final_item_embeddings = None
        self.faiss_index = None

    def _create_adj_matrix(self, df, num_users, num_items):
        # ユーザーとアイテムのインタラクションからグラフのエッジを作成
        user_indices = torch.LongTensor(df["user_idx"].values)
        item_indices = torch.LongTensor(df["item_idx"].values)

        # ユーザー-アイテム、アイテム-ユーザーの両方向のエッジを作成
        edge_index_user_item = torch.stack([user_indices, item_indices + num_users])
        edge_index_item_user = torch.stack([item_indices + num_users, user_indices])
        edge_index = torch.cat([edge_index_user_item, edge_index_item_user], dim=1)

        # 隣接行列をScipyのスパース行列に変換
        adj_matrix_scipy = to_scipy_sparse_matrix(
            edge_index, num_nodes=num_users + num_items
        )

        # 正規化: D^(-1/2) * A * D^(-1/2)
        rowsum = np.array(adj_matrix_scipy.sum(axis=1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = coo_matrix(
            (d_inv_sqrt, (range(len(d_inv_sqrt)), range(len(d_inv_sqrt)))),
            shape=adj_matrix_scipy.shape,
        )
        norm_adj_matrix_scipy = d_mat_inv_sqrt.dot(adj_matrix_scipy).dot(d_mat_inv_sqrt)

        # PyTorchのスパーステンソルに変換
        i = torch.LongTensor([norm_adj_matrix_scipy.row, norm_adj_matrix_scipy.col])
        v = torch.FloatTensor(norm_adj_matrix_scipy.data)
        return torch.sparse_coo_tensor(i, v, norm_adj_matrix_scipy.shape).to(
            self.device
        )

    def fit(self, df: pd.DataFrame):
        self.user_map = {
            user_id: i for i, user_id in enumerate(df["customer_id"].unique())
        }
        self.item_map = {
            item_id: i for i, item_id in enumerate(df["article_id"].unique())
        }
        self.inverse_item_map = {i: item_id for item_id, i in self.item_map.items()}

        df["user_idx"] = df["customer_id"].map(self.user_map)
        df["item_idx"] = df["article_id"].map(self.item_map)

        num_users, num_items = len(self.user_map), len(self.item_map)

        # 正規化された隣接行列を作成
        adj_matrix = self._create_adj_matrix(df, num_users, num_items)

        # BPR用のデータローダーを作成
        dataset = BPRDataset(df, num_items)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # モデルとオプティマイザを初期化
        self.model = LightGCN(
            num_users, num_items, self.embedding_dim, self.n_layers
        ).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        print(f"Start training LightGCN on {self.device}...")
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for users, pos_items, neg_items in tqdm(
                dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}"
            ):
                users, pos_items, neg_items = (
                    users.to(self.device),
                    pos_items.to(self.device),
                    neg_items.to(self.device),
                )

                optimizer.zero_grad()

                # 全ユーザー・アイテムの最終的な埋め込みを取得
                all_user_embeddings, all_item_embeddings = self.model(adj_matrix)

                # バッチ内のユーザー、ポジティブ/ネガティブアイテムの埋め込みを抽出
                user_embs = all_user_embeddings[users]
                pos_item_embs = all_item_embeddings[pos_items]
                neg_item_embs = all_item_embeddings[neg_items]

                # BPR Lossの計算
                pos_scores = torch.sum(user_embs * pos_item_embs, dim=1)
                neg_scores = torch.sum(user_embs * neg_item_embs, dim=1)

                # 正則化項
                reg_loss = (
                    self.reg
                    * (
                        user_embs.norm(2).pow(2)
                        + pos_item_embs.norm(2).pow(2)
                        + neg_item_embs.norm(2).pow(2)
                    )
                    / float(len(users))
                )

                loss = (
                    -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
                    + reg_loss
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(
                f"Epoch {epoch + 1}/{self.epochs}, BPR Loss: {total_loss / len(dataloader):.4f}"
            )

        # 学習後の最終的なアイテム埋め込みを保存
        _, self.final_item_embeddings = self.model(adj_matrix)
        self.final_item_embeddings = self.final_item_embeddings.detach().cpu().numpy()

        # Faissインデックスの構築
        print("Building Faiss index for LightGCN...")
        item_embeddings_normalized = self.final_item_embeddings.copy()
        faiss.normalize_L2(item_embeddings_normalized)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(item_embeddings_normalized)
        print("Faiss index built successfully.")

        breakpoint()

        return self

    def predict(self, user_ids: list[str], top_k: int = 12) -> pd.DataFrame:
        if self.faiss_index is None:
            raise RuntimeError("You must call fit() before predict()")

        print("Starting prediction using LightGCN+Faiss...")
        target_user_indices = [
            self.user_map[uid] for uid in user_ids if uid in self.user_map
        ]

        if not target_user_indices:
            return pd.DataFrame()

        # 学習後の最終的な全ユーザー埋め込みを取得
        all_user_embeddings, _ = self.model.model(
            self.model.adj_matrix
        )  # adj_matrixの渡し方を要検討
        all_user_embeddings = all_user_embeddings.detach().cpu().numpy()
        target_user_embeddings = all_user_embeddings[target_user_indices]

        faiss.normalize_L2(target_user_embeddings)
        D, I = self.faiss_index.search(target_user_embeddings, top_k)

        pred_data = []
        for i, user_idx in enumerate(target_user_indices):
            original_user_id = user_ids[i]
            for j, item_idx in enumerate(I[i]):
                original_item_id = self.inverse_item_map[item_idx]
                score = D[i][j]
                pred_data.append(
                    {
                        "customer_id": original_user_id,
                        "article_id": original_item_id,
                        "score": score,
                    }
                )
        return pd.DataFrame(pred_data)
