import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from models.neumf import NeuMFModel
from schema.config import Config
from schema.models import Metrics
from utils.dataloader import DataLoader
from utils.metrics_calculator import MetricsCalculator


def evaluate(
    pred_df: pd.DataFrame,
    true_items_df: pd.DataFrame,
    metrics_calculator: MetricsCalculator,
) -> Metrics:
    """予測結果に対するメトリクスの計算を行う"""
    pred_df["pred_items"] = pred_df["pred_items"].apply(
        lambda x: [int(item) for item in x.split()]
    )
    merged_df = true_items_df.merge(pred_df, on="customer_id", how="left")
    metrics = metrics_calculator.calc(merged_df, "true_items", "pred_items")
    return metrics


def main():
    config_path = Path(__file__).resolve().parent.parent / "conf" / "main.yaml"
    cfg = Config.load_config(str(config_path))

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    dataset = DataLoader(cfg).load()
    metrics_calculator = MetricsCalculator(cfg)

    valid_true_df = (
        dataset.valid_df.groupby("customer_id")
        .agg(true_items=("article_id", list))
        .reset_index()
    )
    test_true_df = (
        dataset.test_df.groupby("customer_id")
        .agg(true_items=("article_id", list))
        .reset_index()
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with mlflow.start_run(run_name=NeuMFModel.__name__):
            mlflow.log_params(cfg.model_dump())

            model = NeuMFModel(cfg)
            model.fit(dataset.train_df.copy())

            valid_pred_df = model.predict(
                dataset.valid_df[["customer_id"]], num_rec=cfg.eval.num_rec
            )
            valid_metrics = evaluate(valid_pred_df, valid_true_df, metrics_calculator)
            mlflow.log_metrics(
                {f"valid_{k}": v for k, v in valid_metrics.model_dump().items()}
            )

            test_pred_df = model.predict(
                dataset.test_df[["customer_id"]], num_rec=cfg.eval.num_rec
            )
            test_metrics = evaluate(test_pred_df, test_true_df, metrics_calculator)
            mlflow.log_metrics(
                {f"test_{k}": v for k, v in test_metrics.model_dump().items()}
            )


if __name__ == "__main__":
    main()
