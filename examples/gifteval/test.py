import os
from typing import Iterator
from pathlib import Path
import pytest
import yaml

from data_fixed_new_numpy import Dataset
from gift_eval_utils import gift_eval_dataset_iter

from utils import better_gift_eval_dataset_name
from utils import create_test_set_dataframe
from utils import create_training_set_dataframe
from utils import save_dataframe
from utils import get_series_with_gte_missing_target_pct
from utils import filter_data_by_series
from utils import is_multi_series_dataset


@pytest.fixture(scope="module", autouse=True)
def set_gift_eval_path() -> Iterator[None]:
    os.environ["GIFT_EVAL"] = "/home/lyndon.kang/projects/foundation_model_compare/gift_eval_datasets/"
    yield


def test_dataset():
    dataset_info = next(iter(gift_eval_dataset_iter()))
    ds_name = dataset_info["ds_name"]
    term = dataset_info["term"]

    to_univariate = (
        False
        if Dataset(name=dataset_info["ds_name"], term=dataset_info["term"], to_univariate=False).target_dim == 1
        else True
    )
    import pdb
    pdb.set_trace()
    assert True


def test_univariate_mult_series_dataset():
    dataset = Dataset(
        name="electricity/W",
        term="short",
        to_univariate=False,
    )
    df_test = create_test_set_dataframe(dataset)
    df_train = create_training_set_dataframe(dataset)
    import pdb
    pdb.set_trace()
    assert True


def test_univariate_dataset():
    dataset = Dataset(
        name="saugeenday/M",
        term="short",
        to_univariate=False,
    )
    import pdb
    pdb.set_trace()
    df_test = create_test_set_dataframe(dataset)
    df_train = create_training_set_dataframe(dataset)
    import pdb
    pdb.set_trace()
    assert True


def test_multivariate_dataset():
    dataset = Dataset(
        name="ett1/W",
        term="short",
        to_univariate=True,
    )
    import pdb
    pdb.set_trace()
    assert True


def test_create_mbtest_dataset() -> None:
    output_folder_path = Path("/home/lyndon.kang/projects/foundation_model_compare/gift_eval_mbtest_datasets")

    for data_info in gift_eval_dataset_iter(["short"]):
        print(f">>>>>>>>>>>>>>>>>>>>>> {data_info}")
        ds_name = data_info["ds_name"]
        term = data_info["term"]
        to_univariate = (
            False
            if Dataset(name=ds_name, term=term, to_univariate=False).target_dim == 1
            else True
        )
        dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
        training_set_dataframe = create_training_set_dataframe(dataset)
        test_set_dataframe = create_test_set_dataframe(dataset)

        if is_multi_series_dataset(dataset):
            series_to_exclude = get_series_with_gte_missing_target_pct(training_set_dataframe, 0.2)
            training_set_dataframe = filter_data_by_series(training_set_dataframe, series_to_exclude)
            test_set_dataframe = filter_data_by_series(test_set_dataframe, series_to_exclude)

        save_dataframe(training_set_dataframe, output_folder_path / f"{better_gift_eval_dataset_name(ds_name)}_train.csv")
        save_dataframe(test_set_dataframe, output_folder_path / f"{better_gift_eval_dataset_name(ds_name)}_test.csv")


def test_create_mbtest_yaml() -> None:
    mbtest_yaml_content = []
    for data_info in gift_eval_dataset_iter(["short"]):
        print(f">>>>>>>>>>>>>>>>>>>>>> {data_info}")
        ds_name = data_info["ds_name"]
        term = data_info["term"]
        to_univariate = (
            False
            if Dataset(name=ds_name, term=term, to_univariate=False).target_dim == 1
            else True
        )
        dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
        print(f">>>>>>>>>>>>>>>>>>>> {ds_name}")
        path_prefix = "s3://shrink-datasets/gift_eval_short_term_schema"
        path_prefix = "/home/lyndon.kang/projects/foundation_model_compare/gift_eval_short_term_schema"

        target_name = "target"
        datetime_partition_column = "datetime"
        ds_name = better_gift_eval_dataset_name(ds_name)
        mbtest_config = {
            "dataset_name": f"{path_prefix}/{ds_name}_train.csv",
            "metric": "MAE",
            "multiclass": False,
            "partitioning": {"partition_column": datetime_partition_column},
            "prediction_dataset_name": f"{path_prefix}/{ds_name}_test.csv",
            "rtype": "Regression",
            "target": target_name,
            "time_series": {
                "feature_derivation_window_end": 0,
                "feature_derivation_window_start": -dataset.prediction_length,
                "forecast_window_end": dataset.prediction_length,
                "forecast_window_start": 1,
            },
            "use_time_series": True,
        }
        if is_multi_series_dataset(dataset):
            mbtest_config["time_series"].update(
                {"multiseries_id_columns": ["series_id"]}
            )
        mbtest_yaml_content.append(mbtest_config)

        with open(
            "/home/lyndon.kang/projects/foundation_model_compare/mbtest/"
            "gift_eval_short_term_schema_with_local_path.yaml",
            "w"
        ) as output_file:
            yaml.safe_dump(mbtest_yaml_content, output_file)
