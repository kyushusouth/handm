import pandas as pd

from schema.config import Config


class Popularity:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.items_popularity_order = None

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> "Popularity":
        self.items_popularity_order = train_df.groupby("article_id").size().sort_values(ascending=False).index.to_list()
        return self

    def predict(self, row: pd.Series) -> list:
        return self.items_popularity_order[: self.cfg.eval.num_rec]
