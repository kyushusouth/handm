from typing import List

import lightgbm as lgb
import pandas as pd

from rankers.base import BaseRanker
from schema.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class ListwiseRanker(BaseRanker):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.model = lgb.LGBMRanker(**cfg.model.ranker.listwise.params.model_dump())

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame):
        logger.info(f"Fitting {self.__class__.__name__}...")
        self.model.fit(
            X=train_df[self.cfg.features.num_cols + self.cfg.features.cat_cols],
            y=train_df["target"],
            group=train_df.groupby("customer_id").size().values,
            eval_set=[(valid_df[self.cfg.features.num_cols + self.cfg.features.cat_cols], valid_df["target"])],
            eval_group=[valid_df.groupby("customer_id").size().values],
            feature_name=self.cfg.features.num_cols + self.cfg.features.cat_cols,
            categorical_feature=self.cfg.features.cat_cols,
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.cfg.model.ranker.early_stopping_rounds),
                lgb.log_evaluation(period=self.cfg.model.ranker.early_stopping_rounds),
            ],
        )
        return self

    def predict(self, X: pd.DataFrame) -> List[float]:
        return self.model.predict(X)
