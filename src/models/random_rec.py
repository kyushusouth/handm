import random

import pandas as pd

from models.base import BaseModel, check_is_fitted, set_is_fitted_after
from schema.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class RandomRecModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.items = None

    @set_is_fitted_after
    def fit(self, train_df: pd.DataFrame) -> "RandomRecModel":
        self.items = train_df["article_id"].unique().tolist()
        return self

    @check_is_fitted
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        target_cs_ids = X["customer_id"].unique()
        pred_df = pd.DataFrame({"customer_id": target_cs_ids}).assign(
            pred_items=lambda df: [random.sample(self.items, self.cfg.eval.num_rec)] * len(df)
        )
        return pred_df
