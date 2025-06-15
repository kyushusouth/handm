from typing import List

import numpy as np
import pandas as pd

from models.base import BaseModel, check_is_fitted, set_is_fitted_after
from schema.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleModel(BaseModel):
    """
    複数のベースモデルの予測結果を、重み付きで集計して最終的な推薦リストを作成するモデル。
    """

    def __init__(self, cfg: Config, models: List[BaseModel], weights: List[float]):
        super().__init__(cfg)
        if len(models) != len(weights):
            raise ValueError("The number of models and weights must be the same.")

        self.models = models
        self.weights = weights
        self.is_fitted = True

    @set_is_fitted_after
    def fit(self, train_df: pd.DataFrame) -> "EnsembleModel":
        """
        アンサンブルモデルは個別の学習は不要。
        ベースモデルが学習済みであることが前提。
        """
        return self

    @check_is_fitted
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        ベクトル化された操作で、高速にアンサンブル予測を行う。
        """
        target_customer_ids = X["customer_id"].unique()
        all_preds_dfs = [model.predict(X) for model in self.models]

        all_scores = []
        for model, weight, pred_df in zip(self.models, self.weights, all_preds_dfs):
            pred_df = pred_df.loc[(pred_df["pred_items"].apply(lambda x: len(x)) != 0)]
            if pred_df.empty:
                continue

            exploded_df = pred_df.explode("pred_items").rename(columns={"pred_items": "article_id"})
            exploded_df["rank"] = exploded_df.groupby("customer_id").cumcount()
            exploded_df["score"] = (self.cfg.eval.num_rec - exploded_df["rank"]) * weight
            all_scores.append(exploded_df[["customer_id", "article_id", "score"]])

        if not all_scores:
            empty_submission = pd.DataFrame({"customer_id": target_customer_ids})
            empty_submission["pred_items"] = [] * len(target_customer_ids)
            return empty_submission

        all_scores_df = pd.concat(all_scores)
        final_scores = all_scores_df.groupby(["customer_id", "article_id"])["score"].sum().reset_index()

        final_scores = final_scores.sort_values(["customer_id", "score"], ascending=[True, False])
        top_k_df = final_scores.groupby("customer_id").head(self.cfg.eval.num_rec)

        pred_df = pd.DataFrame({"customer_id": target_customer_ids}).merge(
            top_k_df.groupby("customer_id").agg(pred_items=("article_id", list)), on="customer_id", how="left"
        )
        pred_df["pred_items"] = pred_df["pred_items"].apply(lambda x: [] if np.any(pd.isna(x)) else x)

        return pred_df
