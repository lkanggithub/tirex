import os
import pytest
from typing import Iterator

from tirex import ForecastModel, load_model

import pandas as pd
from gift_eval_utils import TiRexGiftEvalWrapper, evaluate_dataset, gift_eval_dataset_iter


@pytest.fixture(scope="module", autouse=True)
def set_gift_eval_path() -> Iterator[None]:
    os.environ["GIFT_EVAL"] = "/home/lkanggithub/projects/foundation_model_compare/gift_eval_datasets/"
    yield


@pytest.fixture
def train_context_length() -> int:
    return 128


def test(train_context_length: int) -> None:
    model: ForecastModel = load_model("NX-AI/TiRex", device="cuda:0")
    wrapped_model = TiRexGiftEvalWrapper(model)
    results = []
    for task in gift_eval_dataset_iter():
        print(f">>>>>>>>>>>>>>>>> {task}")
        task_result = evaluate_dataset(wrapped_model, **task, ds_train_context_length=train_context_length)
        results.append(task_result)
        print(task_result)
    results = pd.DataFrame(results)
    print(results)
