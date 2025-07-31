import os
import pytest
from pathlib import Path
from typing import Iterator
from typing import List

from tirex import load_model

from dr_model_benchmark.common.analysis.entities import ModelScoreMetrics
from dr_model_benchmark.common.analysis.entities import ModelTimeProfiles
from dr_model_benchmark.common.analysis.entities import TestResultV2
from dr_model_benchmark.common.analysis.enums import Partition
from dr_model_benchmark.common.enums import MetricType
from dr_model_benchmark.common.profile.entities import TimeProfile
from dr_model_benchmark.common.profile.utils import TimeProfiler
from dr_model_benchmark.common.profile.enums import TimeProfileType


import pandas as pd
from gift_eval_utils import TiRexGiftEvalWrapper, evaluate_dataset, gift_eval_dataset_iter


@pytest.fixture(scope="module", autouse=True)
def set_gift_eval_path() -> Iterator[None]:
    os.environ["GIFT_EVAL"] = "/home/lkanggithub/projects/foundation_model_compare/gift_eval_datasets/"
    yield


@pytest.fixture
def test_result_output_path() -> Path:
    return Path(
        "/home/lkanggithub/projects/foundation_model_compare/"
        "results_gift_eval_short_term_tirex_gpu_with_time.csv"
    )


def test(test_result_output_path: Path) -> None:
    model = load_model("NX-AI/TiRex", device="cuda:0")
    wrapped_model = TiRexGiftEvalWrapper(model)
    test_results: List[TestResultV2] = []
    gift_eval_dataset_term = "short"
    for task in gift_eval_dataset_iter([gift_eval_dataset_term], only_include_univariate_data=True):
        print(f">>>>>>>>>>>>>>>>> {task}")
        partition = Partition.TEST
        test_time_profile = TimeProfile(partition.name)
        with TimeProfiler(test_time_profile):
            task_result = evaluate_dataset(wrapped_model, **task)
        model_score_metrics = [
            ModelScoreMetrics(
                MetricType.MAE,
                partition=partition,
                score=float(task_result["eval_metrics/MAE[0.5]"]),
            ),
            ModelScoreMetrics(
                MetricType.MAPE,
                partition=partition,
                score=float(task_result["eval_metrics/MAPE[0.5]"]),
            ),
        ]
        model_time_profiles = [
            ModelTimeProfiles(
                TimeProfileType.TOTAL_CLOCK_TIME,
                partition,
                test_time_profile.time_ellipse,
            )
        ]
        test_result = TestResultV2(
            dataset_name=task["ds_name"],
            model_score_metrics=model_score_metrics,
            model_time_profiles=model_time_profiles,
        )
        test_results.append(test_result)
        print(task_result)
    TestResultV2.to_csv(test_results, test_result_output_path)
