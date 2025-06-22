import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.base import BaseModel
from utils.logger import get_logger

logger = get_logger(__name__)


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim=8, mlp_layers=[16, 8]):
        super(NeuMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.mf_dim = mf_dim
        self.mlp_layers = mlp_layers

        self.mf_embedding_user = nn.Embedding(num_users, mf_dim)
        self.mf_embedding_item = nn.Embedding(num_items, mf_dim)
        self.mlp_embedding_user = nn.Embedding(num_users, mlp_layers[0] // 2)
        self.mlp_embedding_item = nn.Embedding(num_items, mlp_layers[0] // 2)

        self.mlp = nn.Sequential()
        for i in range(len(mlp_layers) - 1):
            self.mlp.add_module(
                f"linear_{i}", nn.Linear(mlp_layers[i], mlp_layers[i + 1])
            )
            self.mlp.add_module(f"relu_{i}", nn.ReLU())

        predict_size = mf_dim + mlp_layers[-1]
        self.predict_layer = nn.Linear(predict_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        """
        args:
            user_indices: (B,)
            user_indices: (B,)
        returns:
            predictions: (B,)
        """
        mf_user_latent = self.mf_embedding_user(user_indices)
        mf_item_latent = self.mf_embedding_item(item_indices)
        mf_vector = torch.mul(mf_user_latent, mf_item_latent)

        mlp_user_latent = self.mlp_embedding_user(user_indices)
        mlp_item_latent = self.mlp_embedding_item(item_indices)
        mlp_vector = torch.cat([mlp_user_latent, mlp_item_latent], dim=-1)
        mlp_vector = self.mlp(mlp_vector)

        predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)

        predictions = self.sigmoid(self.predict_layer(predict_vector)).squeeze()
        return predictions


class NeuMFDataset(Dataset):
    """NeuMFモデル学習用のカスタムDatasetクラス"""

    def __init__(self, df: pd.DataFrame):
        self.users = torch.LongTensor(df["user_idx"].values)
        self.items = torch.LongTensor(df["item_idx"].values)
        self.ratings = torch.FloatTensor(np.ones(len(df)))

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class NeuMFModel(BaseModel):
    def __init__(
        self, cfg, mf_dim=8, mlp_layers=[16, 8], epochs=10, batch_size=256, lr=0.001
    ):
        self.cfg = cfg
        self.mf_dim = mf_dim
        self.mlp_layers = mlp_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.index = None
        self.user_map = {}
        self.item_map = {}
        self.inverse_item_map = {}

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

        dataset = NeuMFDataset(df)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        num_users = len(self.user_map)
        num_items = len(self.item_map)
        self.model = NeuMF(num_users, num_items, self.mf_dim, self.mlp_layers).to(
            self.device
        )

        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        print(f"Start training NeuMF on {self.device}...")
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for user_batch, item_batch, label_batch in tqdm(
                dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}"
            ):
                user_batch, item_batch, label_batch = (
                    user_batch.to(self.device),
                    item_batch.to(self.device),
                    label_batch.to(self.device),
                )

                optimizer.zero_grad()
                predictions = self.model(user_batch, item_batch)
                loss = loss_function(predictions, label_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(
                f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(dataloader):.4f}"
            )

        print("Building Faiss index...")
        item_embeddings = self.model.mf_embedding_item.weight.data.cpu().numpy()
        faiss.normalize_L2(item_embeddings)
        self.index = faiss.IndexFlatIP(self.mf_dim)
        self.index.add(item_embeddings)
        # self.index = faiss.IndexIVFFlat(faiss.IndexFlatL2(8), 8, 100)
        # self.index.train(item_embeddings)
        # self.index.add(item_embeddings)
        print("Faiss index built successfully.")

        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        num_rec = kwargs.get("num_rec", self.cfg.eval.num_rec)

        target_customer_ids = X["customer_id"].unique()

        customers_to_predict = []
        customer_indices = []
        for cid in target_customer_ids:
            if cid in self.user_map:
                customers_to_predict.append(cid)
                customer_indices.append(self.user_map[cid])

        if not customers_to_predict:
            return pd.DataFrame({"customer_id": target_customer_ids, "pred_items": ""})

        user_embeddings = self.model.mf_embedding_user.weight.data.cpu().numpy()
        target_user_embeddings = user_embeddings[customer_indices]
        faiss.normalize_L2(target_user_embeddings)
        D, I = self.index.search(target_user_embeddings, num_rec)

        results = []
        for i, customer_id in enumerate(customers_to_predict):
            recommended_indices = I[i]
            recommended_items = [
                str(self.inverse_item_map[idx]) for idx in recommended_indices
            ]
            pred_items_str = " ".join(recommended_items)
            results.append({"customer_id": customer_id, "pred_items": pred_items_str})

        pred_df = pd.DataFrame(results)
        original_customers_df = pd.DataFrame({"customer_id": target_customer_ids})
        pred_df = pd.merge(original_customers_df, pred_df, on="customer_id", how="left")
        pred_df["pred_items"] = pred_df["pred_items"].fillna("")

        return pred_df
