from typing import List, Type

import pandas as pd

from models.base import BaseModel
from models.cooccurrence import CooccurrenceModel
from models.ensemble import EnsembleModel
from models.popularity import PopularityModel
from models.random_rec import RandomRecModel
from models.repurchase import RepurchaseModel
from schema.config import Config
from schema.models import Metrics
from utils.dataloader import DataLoader
from utils.logger import get_logger
from utils.metrics_calculator import MetricsCalculator

logger = get_logger(__name__)


def evaluate(
    pred_df: pd.DataFrame,
    true_items_df: pd.DataFrame,
    metrics_calculator: MetricsCalculator,
) -> Metrics:
    """予測結果に対するメトリクスの計算を行う"""
    merged_df = true_items_df.merge(pred_df, on="customer_id", how="left")
    metrics = metrics_calculator.calc(merged_df, "true_items", "pred_items")
    return metrics


def main():
    cfg = Config.load_config("../conf/main.yaml")

    dataset = DataLoader(cfg).load()
    metrics_calculator = MetricsCalculator(cfg)

    test_true_df = dataset.test_df.groupby("customer_id").agg(true_items=("article_id", list)).reset_index()

    models_to_run: List[Type[BaseModel]] = [
        CooccurrenceModel,
        RepurchaseModel,
        PopularityModel,
        RandomRecModel,
    ]

    all_results = []
    fitted_base_models: List[BaseModel] = []

    for model_class in models_to_run:
        model = model_class(cfg)
        logger.info(f"Running for model: {model.__class__.__name__}")
        model.fit(dataset.train_df.copy())
        pred_df = model.predict(dataset.test_df.copy())
        metrics = evaluate(pred_df, test_true_df, metrics_calculator)
        all_results.append({"model": model.__class__.__name__, **metrics.model_dump()})
        fitted_base_models.append(model)

    ensemble_weights = [1.0, 1.0, 1.0, 1.0]
    model = EnsembleModel(cfg, models=fitted_base_models, weights=ensemble_weights)
    pred_df = model.predict(dataset.test_df.copy())
    metrics = evaluate(pred_df, test_true_df, metrics_calculator)
    all_results.append({"model": model.__class__.__name__, **metrics.model_dump()})

    results_df = pd.DataFrame(all_results)
    print("\n--- Evaluation Results ---")
    print(results_df.round(4))


if __name__ == "__main__":
    main()
