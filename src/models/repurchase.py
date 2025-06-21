from typing import Dict, List

import pandas as pd

from models.base import BaseModel, check_is_fitted, set_is_fitted_after
from schema.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class RepurchaseModel(BaseModel):
    """
    ユーザーが過去に購入したアイテムを推薦するモデル。
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.customer_purchase_history: Dict[str, List[int]] = {}

    @set_is_fitted_after
    def fit(self, train_df: pd.DataFrame):
        """
        学習データから、顧客ごとの購入アイテム履歴を作成する。
        ここでは、時間的に後に出てきたアイテム（より最近のアイテム）がリストの前方にくるように処理する。
        """
        purchase_history = (
            train_df.sort_values(by=["customer_id", "t_dat"], ascending=[True, False])
            .groupby("customer_id")["article_id"]
            .apply(lambda x: x.unique().tolist())
        )
        self.customer_purchase_history = purchase_history.to_dict()
        return self

    @check_is_fitted
    def predict(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        fitで作成した購入履歴辞書から、各顧客への推薦リストを生成する。
        """
        num_rec = kwargs.get("num_rec", self.cfg.eval.num_rec)

        preds = []
        for customer_id in X["customer_id"].unique():
            past_purchases = self.customer_purchase_history.get(customer_id, [])
            pred_items = past_purchases[:num_rec]
            preds.append({"customer_id": customer_id, "pred_items": " ".join([str(item) for item in pred_items])})

        return pd.DataFrame(preds)
