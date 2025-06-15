from datetime import date

import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    """データ関連の設定"""

    train_data_path: str
    chunksize: int
    train_start_date: date
    train_end_date: date
    valid_start_date: date
    valid_end_date: date
    test_start_date: date
    test_end_date: date


class EvalConfig(BaseModel):
    """評価に関わる設定"""

    num_rec: int


class Config(BaseModel):
    """プロジェクト全体のConfigを管理するトップレベルクラス"""

    seed: int
    log_config_path: str
    data: DataConfig
    eval: EvalConfig

    model_config = {"frozen": True}

    @classmethod
    def load_config(cls, config_path: str) -> "Config":
        """YAMLファイルを読み込み、Pydanticモデルのインスタンスを返す"""
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)
