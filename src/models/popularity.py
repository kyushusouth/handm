import pandas as pd

from models.base import BaseModel, check_is_fitted, set_is_fitted_after
from schema.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class PopularityModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.items_popularity_order = None

    @set_is_fitted_after
    def fit(self, train_df: pd.DataFrame) -> "PopularityModel":
        self.items_popularity_order = train_df.groupby("article_id").size().sort_values(ascending=False).index.to_list()
        return self

    @check_is_fitted
    def predict(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        num_rec = kwargs.get("num_rec", self.cfg.eval.num_rec)

        pred_df = pd.DataFrame({"customer_id": X["customer_id"].unique()}).assign(
            pred_items=lambda df: [self.items_popularity_order[:num_rec]] * len(df)
        )
        return pred_df
