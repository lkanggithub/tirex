import os
import pytest
from pathlib import Path
from typing import Iterator

from tirex import ForecastModel, load_model

import pandas as pd
from gift_eval_utils import TiRexGiftEvalWrapper, evaluate_dataset, gift_eval_dataset_iter


@pytest.fixture(scope="module", autouse=True)
def set_gift_eval_path() -> Iterator[None]:
    os.environ["GIFT_EVAL"] = "/home/lkanggithub/projects/foundation_model_compare/gift_eval_datasets/"
    yield


@pytest.fixture
def test_result_output_path() -> Path:
    return Path("/home/lkanggithub/projects/foundation_model_compare/gift_eval_short_term_results.csv")


def test(test_result_output_path: Path) -> None:
    model: ForecastModel = load_model("NX-AI/TiRex", device="cuda:0")
    wrapped_model = TiRexGiftEvalWrapper(model)
    results = []
    gift_eval_dataset_term = "short"
    for task in gift_eval_dataset_iter([gift_eval_dataset_term]):
        print(f">>>>>>>>>>>>>>>>> {task}")
        task_result = evaluate_dataset(wrapped_model, **task)
        results.append(task_result)
        print(task_result)
    results = pd.DataFrame(results)
    results.to_csv(test_result_output_path, index=False)
