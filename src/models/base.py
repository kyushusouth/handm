import functools
import logging
from abc import ABC, abstractmethod

import pandas as pd

from schema.config import Config


def set_is_fitted_after(func):
    """fitメソッドの実行後にis_fittedをTrueにし、ログを出力するデコレータ"""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self.is_fitted = True
        logger = logging.getLogger(self.__class__.__module__)
        logger.info(f"{self.__class__.__name__} has been fitted.")
        return result

    return wrapper


def check_is_fitted(func):
    """predictメソッドの実行前にis_fittedをチェックするデコレータ"""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} has not been fitted.")
        return func(self, *args, **kwargs)

    return wrapper


class BaseModel(ABC):
    """全ての推薦モデルが継承するベースクラス"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.is_fitted = False

    @abstractmethod
    def fit(self, train_df: pd.DataFrame):
        """
        学習データを使ってモデルの状態を更新（学習）する
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        予測対象の顧客情報などが入ったDataFrameを受け取り、予測結果を付与したDataFrameを返す
        """
        raise NotImplementedError
