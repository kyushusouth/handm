import os
from typing import Tuple

import pandas as pd

from schema.config import Config
from schema.models import Dataset
from utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def load(self) -> Dataset:
        """Datasetのインスタンスを返す"""
        logger.info("Loading Dataset ...")
        data_df = self._load()
        train_df, valid_df, test_df = self._split_data(data_df)
        return Dataset(train_df=train_df, valid_df=valid_df, test_df=test_df)

    def _load(self) -> pd.DataFrame:
        """データを読み込んで返す"""
        start_date = pd.to_datetime(self.cfg.data.train_start_date)
        end_date = pd.to_datetime(self.cfg.data.test_end_date)
        filtered_chunks = []
        for chunk in pd.read_csv(os.path.expanduser(self.cfg.data.train_data_path), chunksize=self.cfg.data.chunksize):
            chunk["t_dat"] = pd.to_datetime(chunk["t_dat"])
            filtered_chunk = chunk.query("@start_date <= t_dat <= @end_date")
            if not filtered_chunk.empty:
                filtered_chunks.append(filtered_chunk)
        data_df = pd.concat(filtered_chunks)
        return data_df

    def _split_data(self, data_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """データを学習、検証、テストに分割して返す"""
        train_start_date = pd.to_datetime(self.cfg.data.train_start_date)
        train_end_date = pd.to_datetime(self.cfg.data.train_end_date)
        valid_start_date = pd.to_datetime(self.cfg.data.valid_start_date)
        valid_end_date = pd.to_datetime(self.cfg.data.valid_end_date)
        test_start_date = pd.to_datetime(self.cfg.data.test_start_date)
        test_end_date = pd.to_datetime(self.cfg.data.test_end_date)
        train_df = data_df.query("@train_start_date <= t_dat <= @train_end_date")
        valid_df = data_df.query("@valid_start_date <= t_dat <= @valid_end_date")
        test_df = data_df.query("@test_start_date <= t_dat <= @test_end_date")
        return train_df, valid_df, test_df
