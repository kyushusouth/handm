import pandas as pd
from pydantic import BaseModel


class Dataset(BaseModel):
    """データセット"""

    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame

    model_config = {"arbitrary_types_allowed": True, "frozen": True}


class Metrics(BaseModel):
    """評価指標"""

    precision: float
    recall: float
    f1: float
    mrr: float
    map: float
    ndcg: float

    model_config = {"frozen": True}

    def __repr__(self) -> str:
        """
        各メトリクスを改行区切りで表示する
        """
        lines = [f"{key}={value:.3f}" for key, value in self.model_dump().items()]
        return f"{self.__class__.__name__}(\n " + ",\n ".join(lines) + "\n)"

    def __str__(self):
        return self.__repr__()
