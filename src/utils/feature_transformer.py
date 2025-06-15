from typing import Dict, Tuple

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

    def _convert_cat_features(self, features_df: pd.DataFrame):
        for col in self.cfg.features.cat_cols:
            if col not in features_df:
                continue
            encoder = self.label_encoders[col]
            filled_series = features_df[col].astype(str).fillna("missing")
            known_classes = encoder.classes_
            filled_series = filled_series.apply(lambda x: x if x in known_classes else "missing")
            features_df[col] = encoder.transform(filled_series)
            logger.info(f"finish encoding {col}")

    def transform(
        self,
        train_transactions_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        articles_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """学習済みのエンコーダーを使って、特徴量を作成・変換する"""
        logger.info("Transforming data into features...")

        customer_agg = (
            train_transactions_df.groupby("customer_id")
            .agg(mean_price=("price", "mean"), num_purchases=("article_id", "count"))
            .reset_index()
        )

        logger.info("finish customer_agg")

        last_date = train_transactions_df["t_dat"].max()
        recent_transactions = train_transactions_df[train_transactions_df["t_dat"] > last_date - pd.DateOffset(days=14)]
        article_agg = recent_transactions.groupby("article_id").agg(recent_sales=("customer_id", "count")).reset_index()

        logger.info("finish article_agg")

        customer_feature_df = customers_df.merge(customer_agg, on="customer_id", how="left")
        customer_feature_df["num_purchases"] = customer_feature_df["num_purchases"].fillna(0)
        article_feature_df = articles_df.merge(article_agg, on="article_id", how="left")
        article_feature_df["recent_sales"] = article_feature_df["recent_sales"].fillna(0)

        self._convert_cat_features(customer_feature_df)
        self._convert_cat_features(article_feature_df)

        logger.info(f"Feature transformation complete. {customer_feature_df.shape=}, {article_feature_df.shape=}")
        return customer_feature_df, article_feature_df
