import random

import pandas as pd

from schema.config import Config


class RandomRec:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.items = None

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> "RandomRec":
        self.items = train_df["article_id"].unique().tolist()
        return self

    def predict(self, row: pd.Series) -> list:
        return random.sample(self.items, self.cfg.eval.num_rec)
