from typing import List, Type

import pandas as pd

from models.base import BaseModel
from models.popularity import PopularityModel
from models.random_rec import RandomRecModel
from schema.config import Config
from schema.models import Metrics
from utils.dataloader import DataLoader
from utils.logger import get_logger
from utils.metrics_calculator import MetricsCalculator

logger = get_logger(__name__)


def evaluate(
    model: BaseModel,
    eval_df: pd.DataFrame,
    true_items_df: pd.DataFrame,
    metrics_calculator: MetricsCalculator,
) -> Metrics:
    """
    単一のモデルを評価し、メトリクスを計算する関数。
    """
    logger.info(f"--- Evaluating {model.__class__.__name__} ---")
    pred_df = model.predict(eval_df)
    merged_df = true_items_df.merge(pred_df, on="customer_id", how="left")
    metrics = metrics_calculator.calc(merged_df, "true_items", "pred_items")
    return metrics


def main():
    cfg = Config.load_config("../conf/main.yaml")

    dataset = DataLoader(cfg).load()
    metrics_calculator = MetricsCalculator(cfg)

    valid_true_df = dataset.valid_df.groupby("customer_id").agg(true_items=("article_id", list)).reset_index()
    test_true_df = dataset.test_df.groupby("customer_id").agg(true_items=("article_id", list)).reset_index()

    models_to_run: List[Type[BaseModel]] = [
        PopularityModel,
        RandomRecModel,
    ]

    all_results = []

    for model_class in models_to_run:
        model = model_class(cfg)
        logger.info(f"===== Running for model: {model.__class__.__name__} =====")

        model.fit(dataset.train_df.copy())

        val_metrics = evaluate(model, dataset.valid_df.copy(), valid_true_df, metrics_calculator)
        val_result = {"model": model.__class__.__name__, "split": "validation", **val_metrics.model_dump()}
        all_results.append(val_result)

        test_metrics = evaluate(model, dataset.test_df.copy(), test_true_df, metrics_calculator)
        test_result = {"model": model.__class__.__name__, "split": "test", **test_metrics.model_dump()}
        all_results.append(test_result)

    results_df = pd.DataFrame(all_results)
    logger.info("===== All Model Results Summary =====")

    print("\n--- Evaluation Results ---")
    print(results_df.round(4))


if __name__ == "__main__":
    main()
