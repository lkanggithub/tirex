import os
from typing import Iterator
from pathlib import Path
import pytest

from data_fixed_new_numpy import Dataset
from gift_eval_utils import gift_eval_dataset_iter

from utils import better_gift_eval_dataset_name
from utils import create_test_set_dataframe
from utils import create_training_set_dataframe
from utils import save_dataframe


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

    for data_info in gift_eval_dataset_iter():
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

        save_dataframe(training_set_dataframe, output_folder_path / f"{better_gift_eval_dataset_name(ds_name)}_train.csv")
        save_dataframe(test_set_dataframe, output_folder_path / f"{better_gift_eval_dataset_name(ds_name)}_test.csv")


"""
        input_transform = AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
        ) + InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=TestSplitSampler(),
            past_length=context_length,
            future_length=prediction_length,
            time_series_fields=[FieldName.OBSERVED_VALUES],
        )
        
        
        TotoSampleForecastGenerator(SampleForecastGenerator)
        from gluonts.model.forecast_generator import SampleForecastGenerator

"""