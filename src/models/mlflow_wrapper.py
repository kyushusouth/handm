import pickle
from typing import Any, Dict, Optional

import mlflow
import pandas as pd


class MlflowModelWrapper(mlflow.pyfunc.PythonModel):
    """
    pickleで保存されたカスタムモデルをロードし、
    MLflowのpyfunc形式で扱えるようにするラッパークラス。
    """

    def load_context(self, context):
        """
        MLflowがモデルをロードする際に呼び出す。
        関連付けられたpickleファイルをロードする。
        """
        with open(context.artifacts["model_pickle_path"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        MLflowの標準的なpredictインターフェース。
        内部で、ロードしたカスタムモデルのpredictメソッドを呼び出す。
        """
        if params is None:
            params = {}
        return self.model.predict(model_input, **params)
