from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from schema.config import Config


class BaseRanker(ABC):
    """全てのリランキングモデルが継承するベースクラス"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> List[float]:
        raise NotImplementedError
