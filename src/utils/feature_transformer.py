from typing import Dict

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from schema.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureTransformer:
    """
    特徴量エンジニアリングとエンコーディングの状態を管理するクラス。
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def fit(self, articles_df: pd.DataFrame, customers_df: pd.DataFrame) -> "FeatureTransformer":
        """学習用データを使ったLabelEncoderの学習"""
        logger.info("Fitting FeatureTransformer...")
        for col in self.cfg.features.cat_cols:
            if col in articles_df.columns:
                encoder = LabelEncoder()
                encoder.fit(articles_df[col].astype("str").fillna("missing"))
                self.label_encoders[col] = encoder
            elif col in customers_df.columns:
                encoder = LabelEncoder()
                encoder.fit(customers_df[col].astype("str").fillna("missing"))
                self.label_encoders[col] = encoder

        logger.info("LabelEncoders have been fitted.")
        return self

    def transform(
        self,
        transactions_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        articles_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """学習済みのエンコーダーを使って、特徴量を作成・変換する"""
        logger.info("Transforming data into features...")

        customer_agg = (
            transactions_df.groupby("customer_id")
            .agg(mean_price=("price", "mean"), num_purchases=("article_id", "count"))
            .reset_index()
        )

        logger.info("finish customer_agg")

        last_date = transactions_df["t_dat"].max()
        recent_transactions = transactions_df[transactions_df["t_dat"] > last_date - pd.DateOffset(days=14)]
        article_agg = recent_transactions.groupby("article_id").agg(recent_sales=("customer_id", "count")).reset_index()

        logger.info("finish article_agg")

        features_df = transactions_df.merge(customers_df, on="customer_id", how="left")
        features_df = features_df.merge(articles_df, on="article_id", how="left")
        features_df = features_df.merge(customer_agg, on="customer_id", how="left")
        features_df = features_df.merge(article_agg, on="article_id", how="left")
        features_df["recent_sales"] = features_df["recent_sales"].fillna(0)

        logger.info("finish merge")

        for col in self.cfg.features.cat_cols:
            if col not in features_df:
                raise ValueError(f"{col} was not included in features_df")
            encoder = self.label_encoders[col]
            filled_series = features_df[col].astype(str).fillna("missing")
            known_classes = encoder.classes_
            filled_series = filled_series.apply(lambda x: x if x in known_classes else "missing")
            features_df[col] = encoder.transform(filled_series)

            logger.info(f"finish encoding {col}")

        features_df = features_df[
            ["article_id", "customer_id"] + self.cfg.features.cat_cols + self.cfg.features.num_cols
        ]

        logger.info(f"Feature transformation complete. Shape: {features_df.shape}")
        return features_df
