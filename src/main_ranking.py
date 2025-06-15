from typing import List, Type

import mlflow
import pandas as pd

from models.base import BaseModel
from models.cooccurrence import CooccurrenceModel
from models.ensemble import EnsembleModel
from models.popularity import PopularityModel
from models.random_rec import RandomRecModel
from models.repurchase import RepurchaseModel
from rankers.base import BaseRanker
from rankers.listwise import ListwiseRanker
from rankers.pointwise import PointwiseRanker
from schema.config import Config
from schema.models import Metrics
from utils.dataloader import DataLoader
from utils.feature_transformer import FeatureTransformer
from utils.logger import get_logger
from utils.metrics_calculator import MetricsCalculator

logger = get_logger(__name__)


def evaluate(
    eval_df: pd.DataFrame,
    model: BaseRanker,
    cfg: Config,
    true_items_df: pd.DataFrame,
    metrics_calculator: MetricsCalculator,
) -> Metrics:
    """評価データに対するメトリクスの計算"""
    pred_df = (
        eval_df.assign(pred=lambda df: model.predict(df[cfg.features.cat_cols + cfg.features.num_cols]))
        .sort_values(["customer_id", "pred"], ascending=[True, False])
        .groupby(["customer_id"])
        .agg(pred_items=("article_id", list))
    )
    pred_df["pred_items"] = pred_df["pred_items"].apply(lambda x: x[: cfg.eval.num_rec])
    merged_df = true_items_df.merge(pred_df, on="customer_id", how="left")
    metrics = metrics_calculator.calc(merged_df, "true_items", "pred_items")
    return metrics


def generate_candidates(
    cfg: Config,
    target_customer_ids: List[str],
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    """複数の候補生成モデルを使い、候補リストを作成する"""
    candidate_models: List[Type[BaseModel]] = [
        CooccurrenceModel,
        RepurchaseModel,
        PopularityModel,
        RandomRecModel,
    ]
    fitted_base_models: List[BaseModel] = []
    all_candidates = []

    for model_class in candidate_models:
        model = model_class(cfg).fit(transactions_df)
        logger.info(f"Running for model: {model.__class__.__name__}")
        preds_df = model.predict(pd.DataFrame({"customer_id": target_customer_ids}))
        candidates = preds_df.explode("pred_items").rename(columns={"pred_items": "article_id"})
        all_candidates.append(candidates)
        fitted_base_models.append(model)

    ensemble_weights = [1.0, 1.0, 1.0, 1.0]
    model = EnsembleModel(cfg, models=fitted_base_models, weights=ensemble_weights)
    preds_df = model.predict(pd.DataFrame({"customer_id": target_customer_ids}))
    candidates = preds_df.explode("pred_items").rename(columns={"pred_items": "article_id"})
    all_candidates.append(candidates)

    candidates_df = pd.concat(all_candidates).drop_duplicates()
    logger.info(f"Generated {len(candidates_df)} candidates for {len(target_customer_ids)} customers.")
    return candidates_df


def format_data_for_ranker(
    transactions_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    customer_feature_df: pd.DataFrame,
    article_feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """ランク学習のためにデータを整形"""
    return (
        candidates_df.merge(customer_feature_df, on="customer_id", how="left")
        .merge(article_feature_df, on="article_id", how="left")
        .merge(
            transactions_df[["customer_id", "article_id"]].assign(target=1),
            on=["customer_id", "article_id"],
            how="left",
        )
        .fillna({"target": 0})
        .sort_values("customer_id")
    )


def main():
    cfg = Config.load_config("../conf/main.yaml")

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    dataset = DataLoader(cfg).load()
    feature_transformer = FeatureTransformer(cfg)
    metrics_calculator = MetricsCalculator(cfg)

    valid_true_df = dataset.valid_df.groupby("customer_id").agg(true_items=("article_id", list)).reset_index()
    test_true_df = dataset.test_df.groupby("customer_id").agg(true_items=("article_id", list)).reset_index()

    logger.info("Phase 1: Candidate Generation")
    train_candidates_df = generate_candidates(cfg, dataset.train_df["customer_id"].unique(), dataset.train_df)
    valid_candidates_df = generate_candidates(cfg, dataset.valid_df["customer_id"].unique(), dataset.train_df)
    test_candidates_df = generate_candidates(cfg, dataset.test_df["customer_id"].unique(), dataset.train_df)

    logger.info("Phase 2: Feature Engineering")
    feature_transformer.fit(dataset.articles_df, dataset.customers_df)
    customer_feature_df, article_feature_df = feature_transformer.transform(
        dataset.train_df, dataset.customers_df, dataset.articles_df
    )

    logger.info("Phase 3: Format Data")
    train_df = format_data_for_ranker(
        dataset.train_df.copy(), train_candidates_df, customer_feature_df, article_feature_df
    )
    valid_df = format_data_for_ranker(
        dataset.valid_df.copy(), valid_candidates_df, customer_feature_df, article_feature_df
    )
    test_df = format_data_for_ranker(
        dataset.test_df.copy(), test_candidates_df, customer_feature_df, article_feature_df
    )

    logger.info("Phase 4: Training and Evaluation")
    rankers_to_run: List[Type[BaseRanker]] = [
        PointwiseRanker,
        ListwiseRanker,
    ]

    for ranker_class in rankers_to_run:
        with mlflow.start_run(run_name=ranker_class.__name__):
            logger.info(f"Running for model: {ranker_class.__name__}")
            mlflow.log_params(cfg.model_dump())

            model = ranker_class(cfg)
            model.fit(train_df, valid_df)

            valid_metrics = evaluate(valid_df, model, cfg, valid_true_df, metrics_calculator)
            mlflow.log_metrics({f"valid_{k}": v for k, v in valid_metrics.model_dump().items()})

            test_metrics = evaluate(test_df, model, cfg, test_true_df, metrics_calculator)
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.model_dump().items()})


if __name__ == "__main__":
    main()
