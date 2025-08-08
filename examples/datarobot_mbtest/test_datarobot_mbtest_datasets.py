
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
from dr_model_benchmark.common.profile.entities import Seconds
from dr_model_benchmark.common.profile.entities import TimeProfile
from dr_model_benchmark.common.profile.utils import TimeProfiler
from dr_model_benchmark.common.profile.enums import TimeProfileType

from data import TestDataset


from utils import TiRexGiftEvalWrapper, evaluate_dataset


@pytest.fixture(scope="module", autouse=True)
def set_gift_eval_path() -> Iterator[None]:
    os.environ["GIFT_EVAL"] = "/home/lkanggithub/projects/foundation_model_compare/gift_eval_datasets/"
    yield


@pytest.fixture
def test_result_output_path() -> Path:
    return Path(
        "/home/lkanggithub/projects/foundation_model_compare/"
        "results_datarobot_mbtest_datasets_tirex_gpu_with_unit_time.csv"
    )


def test(test_result_output_path: Path) -> None:
    model = load_model("NX-AI/TiRex", device="cuda:0")
    wrapped_model = TiRexGiftEvalWrapper(model)
    test_results: List[TestResultV2] = []
    datarobot_mbtest_yaml_path = Path(
        "/home/lkanggithub/projects/foundation_model_compare/"
        "custom_data_no_pii_ts_with_lab_machine_path.yaml"
    )
    time_series_frequence = "D"
    test_datasets = TestDataset.create_from_datarobot_mbtest_yaml(
        datarobot_mbtest_yaml_path, time_series_frequence, True,
    )
    for test_dataset in test_datasets:
        print(f">>>>>>>>>>>>>>>>> {test_dataset.name}")
        partition = Partition.TEST
        test_time_profile = TimeProfile(partition.name)
        with TimeProfiler(test_time_profile):
            task_result = evaluate_dataset(wrapped_model, test_dataset)
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
                Seconds(
                    test_time_profile.time_ellipse.to_float() / test_dataset.num_of_forecast_points
                ),
            )
        ]
        test_result = TestResultV2(
            dataset_name=test_dataset.name,
            model_score_metrics=model_score_metrics,
            model_time_profiles=model_time_profiles,
        )
        test_results.append(test_result)
        print(task_result)
    TestResultV2.to_csv(test_results, test_result_output_path)
