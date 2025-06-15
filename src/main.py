import pandas as pd

from models.popularity import Popularity
from models.random_rec import RandomRec
from schema.config import Config
from utils.dataloader import DataLoader
from utils.metrics_calculator import MetricsCalculator


def main():
    cfg = Config.load_config("../conf/main.yaml")
    dataset = DataLoader(cfg).load()
    metrics_calculator = MetricsCalculator(cfg)
    model = Popularity(cfg)

    model.fit(dataset.train_df.copy(), dataset.valid_df.copy())

    eval_df = dataset.test_df.copy().groupby("customer_id").agg(items=("article_id", list))
    eval_df = eval_df.merge(
        eval_df.apply(lambda row: pd.Series({"pred_items": model.predict(row)}), axis=1), on="customer_id", how="left"
    )

    metrics = metrics_calculator.calc(eval_df, "items", "pred_items")
    print(metrics)


if __name__ == "__main__":
    main()
