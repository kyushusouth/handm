from collections import defaultdict
from typing import Dict

import pandas as pd

from models.base import BaseModel, check_is_fitted, set_is_fitted_after
from schema.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class CooccurrenceModel(BaseModel):
    """
    ユーザーが最近購入したアイテムと、一緒によく購入されているアイテム（共起アイテム）を推薦するモデル。
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cooccurrence_dict: Dict[int, Dict[int, int]] = defaultdict(dict)
        self.last_purchases: pd.Series = pd.Series()

    @set_is_fitted_after
    def fit(self, train_df: pd.DataFrame) -> "CooccurrenceModel":
        """
        学習データから、共起行列と各ユーザーの最終購入アイテムを計算して保持する。
        """
        session_df = train_df.copy()
        session_df["session_id"] = session_df["customer_id"] + "_" + session_df["t_dat"].dt.strftime("%Y%m%d")
        merged_df = session_df.merge(session_df, on="session_id", suffixes=("_x", "_y"))
        co_matrix = merged_df.query("article_id_x != article_id_y")
        co_counts = co_matrix.groupby(["article_id_x", "article_id_y"]).size()

        for (aid_x, aid_y), count in co_counts.items():
            self.cooccurrence_dict[aid_x][aid_y] = count

        self.last_purchases = train_df.loc[train_df.groupby("customer_id")["t_dat"].idxmax()]
        self.last_purchases = self.last_purchases.set_index("customer_id")["article_id"]
        return self

    @check_is_fitted
    def predict(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        各顧客の最終購入アイテムに基づき、共起アイテムを推薦する。
        """
        num_rec = kwargs.get("num_rec", self.cfg.eval.num_rec)

        preds = []
        for cid in X["customer_id"].unique():
            pred_items = []
            last_aid = self.last_purchases.get(cid)

            if last_aid and last_aid in self.cooccurrence_dict:
                co_items = self.cooccurrence_dict[last_aid]
                pred_items = sorted(co_items, key=co_items.get, reverse=True)[:num_rec]

            preds.append({"customer_id": cid, "pred_items": " ".join([str(item) for item in pred_items])})

        return pd.DataFrame(preds)
