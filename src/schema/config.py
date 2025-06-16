from datetime import date
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel


class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str


class DataConfig(BaseModel):
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


class ConcurrenceModelConfig(BaseModel):
    model_uri: Optional[str] = None


class RepurchaseModelConfig(BaseModel):
    model_uri: Optional[str] = None


class PopularityModelConfig(BaseModel):
    model_uri: Optional[str] = None


class RandomModelConfig(BaseModel):
    model_uri: Optional[str] = None


class EnsembleModelConfig(BaseModel):
    weights: List[float]
    model_uri: Optional[str] = None


class IMFModelParams(BaseModel):
    factors: int
    regularization: float
    alpha: float
    iterations: int
    random_state: int


class IMFModelConfig(BaseModel):
    params: IMFModelParams
    run_id: Optional[str] = None
    model_uri: Optional[str] = None


class RankerCommonParams(BaseModel):
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


class PointwiseRankerParams(RankerCommonParams):
    objective: Literal["binary"]
    metric: Literal["auc", "binary_logloss"]
    scale_pos_weight: Optional[float] = None


class ListwiseRankerParams(RankerCommonParams):
    objective: Literal["lambdarank"]
    metric: Literal["map", "ndcg"]
    eval_at: List[int]
    lambdarank_truncation_level: int


class PointwiseRankerConfig(BaseModel):
    params: PointwiseRankerParams
    model_uri: Optional[str] = None


class ListwiseRankerConfig(BaseModel):
    params: ListwiseRankerParams
    model_uri: Optional[str] = None


class RankerConfig(BaseModel):
    pointwise: PointwiseRankerConfig
    listwise: ListwiseRankerConfig
    early_stopping_rounds: int
    num_candidates: int


class ModelConfig(BaseModel):
    cooccurrence: ConcurrenceModelConfig
    repurchase: RepurchaseModelConfig
    popularity: PopularityModelConfig
    random: RandomModelConfig
    ensemble: EnsembleModelConfig
    imf: IMFModelConfig
    ranker: RankerConfig


class FeaturesConfig(BaseModel):
    num_cols: List[str]
    cat_cols: List[str]


class EvalConfig(BaseModel):
    num_rec: int


class Config(BaseModel):
    seed: int
    log_config_path: str
    mlflow: MLflowConfig
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
