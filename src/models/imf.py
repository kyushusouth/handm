from typing import Dict

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from models.base import BaseModel, check_is_fitted, set_is_fitted_after
from schema.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class IMFModel(BaseModel):
    """
    行列分解 (ALS) を用いて、協調フィルタリングによる推薦を行うモデル。
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.model = AlternatingLeastSquares(**self.cfg.model.imf.params.model_dump())
        self.customer_id2index: Dict[int, int] = None
        self.customer_index2id: Dict[int, int] = None
        self.article_index2id: Dict[int, int] = None
        self.user_item_matrix: csr_matrix = None
        self.user_factors_df: pd.DataFrame = None
        self.item_factors_df: pd.DataFrame = None

    @set_is_fitted_after
    def fit(self, train_df: pd.DataFrame):
        logger.info(f"Fitting {self.__class__.__name__}...")

        unique_customer_ids = sorted(train_df["customer_id"].unique())
        unique_article_ids = sorted(train_df["article_id"].unique())
        self.customer_id2index = dict(zip(unique_customer_ids, range(len(unique_customer_ids))))
        self.customer_index2id = {v: k for k, v in self.customer_id2index.items()}
        article_id2index = dict(zip(unique_article_ids, range(len(unique_article_ids))))
        self.article_index2id = {v: k for k, v in article_id2index.items()}

        self.user_item_matrix = csr_matrix((len(unique_customer_ids), len(unique_article_ids)))
        for row in train_df.itertuples():
            customer_index = self.customer_id2index[row.customer_id]
            article_index = article_id2index[row.article_id]
            self.user_item_matrix[customer_index, article_index] = 1.0 * self.cfg.model.imf.params.alpha

        self.model.fit(self.user_item_matrix)

        user_factors = self.model.user_factors
        user_ids = [self.customer_index2id[i] for i in range(len(user_factors))]
        self.user_factors_df = pd.DataFrame(user_factors, index=user_ids)
        self.user_factors_df.columns = [f"user_emb_{i}" for i in range(user_factors.shape[1])]
        self.user_factors_df = self.user_factors_df.reset_index(names="customer_id")

        item_factors = self.model.item_factors
        item_ids = [self.article_index2id[i] for i in range(len(item_factors))]
        self.item_factors_df = pd.DataFrame(item_factors, index=item_ids)
        self.item_factors_df.columns = [f"item_emb_{i}" for i in range(item_factors.shape[1])]
        self.item_factors_df = self.item_factors_df.reset_index(names="article_id")

        return self

    @check_is_fitted
    def predict(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        num_rec = kwargs.get("num_rec", self.cfg.eval.num_rec)

        target_customer_indices = []
        for customer_id in X["customer_id"].unique():
            if customer_id in self.customer_id2index:
                target_customer_indices.append(self.customer_id2index[customer_id])

        all_article_indices, scores = self.model.recommend(
            userid=target_customer_indices,
            user_items=self.user_item_matrix[target_customer_indices],
            N=num_rec,
            filter_already_liked_items=True,
            recalculate_user=False,
        )

        preds = []
        for customer_index, article_indices in zip(target_customer_indices, all_article_indices):
            customer_id = self.customer_index2id[customer_index]
            pred_items = [self.article_index2id[article_index] for article_index in article_indices]
            preds.append({"customer_id": customer_id, "pred_items": pred_items})

        pred_df = pd.DataFrame(preds)
        pred_df = pd.DataFrame({"customer_id": X["customer_id"].unique()}).merge(pred_df, on="customer_id", how="left")
        pred_df["pred_items"] = pred_df["pred_items"].apply(lambda x: [] if np.any(pd.isna(x)) else x)

        return pred_df
