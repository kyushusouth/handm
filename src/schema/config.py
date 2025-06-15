from datetime import date
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    """データ関連の設定"""

    train_data_path: str
    customer_data_path: str
    articles_data_path: str
    chunksize: int
    train_start_date: date
    train_end_date: date
    valid_start_date: date
    valid_end_date: date
    test_start_date: date
    test_end_date: date


class BaseRankerParams(BaseModel):
    """PointwiseとListwiseで共通するLGBMのパラメータ"""

    boosting_type: str
    verbosity: int
    seed: int
    n_estimators: int
    learning_rate: float
    num_leaves: int
    max_depth: int
    reg_alpha: float
    reg_lambda: float
    colsample_bytree: float
    subsample: float
    subsample_freq: int


class PointwiseRankerParams(BaseRankerParams):
    """Pointwise Rankerに特有のパラメータ"""

    objective: Literal["binary"]
    metric: Literal["auc", "binary_logloss"]
    scale_pos_weight: Optional[float] = None


class ListwiseRankerParams(BaseRankerParams):
    """Listwise Rankerに特有のパラメータ"""

    objective: Literal["lambdarank"]
    metric: Literal["map", "ndcg"]
    eval_at: List[int]
    lambdarank_truncation_level: int


class RankerConfig(BaseModel):
    """ランキングモデル全体の設定"""

    pointwise: PointwiseRankerParams
    listwise: ListwiseRankerParams
    early_stopping_rounds: int


class ModelConfig(BaseModel):
    """モデル全般の設定"""

    ranker: RankerConfig


class FeaturesConfig(BaseModel):
    """特徴量の設定"""

    num_cols: List[str]
    cat_cols: List[str]


class EvalConfig(BaseModel):
    """評価に関わる設定"""

    num_rec: int


class Config(BaseModel):
    """プロジェクト全体のConfigを管理するトップレベルクラス"""

    seed: int
    log_config_path: str
    data: DataConfig
    model: ModelConfig
    features: FeaturesConfig
    eval: EvalConfig

    model_config = {"frozen": True}

    @classmethod
    def load_config(cls, config_path: str) -> "Config":
        """YAMLファイルを読み込み、Pydanticモデルのインスタンスを返す"""
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)
