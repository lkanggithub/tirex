from pathlib import Path
from typing import List

import pandas as pd
from gluonts.dataset import DataEntry
from gluonts.dataset.util import period_index

from data_fixed_new_numpy import Dataset


def is_multi_series_dataset(dataset: Dataset) -> bool:
    return len(dataset.full_dataset) >= 2


def create_dataframe_from_one_data_entry(
    data_entry: DataEntry, is_multi_series: bool,
) -> pd.DataFrame:
    data = {
        "datetime": period_index(data_entry, freq=data_entry["freq"]),
        "target": data_entry["target"],
    }
    if is_multi_series:
        data.update({"series_id": data_entry["item_id"]})
    return pd.DataFrame(data)


def create_training_set_dataframe_from_one_entry(
    data_entry: DataEntry, is_multi_series: bool, offset: int,
) -> pd.DataFrame:
    data_frame = create_dataframe_from_one_data_entry(data_entry, is_multi_series)
    return data_frame[: offset+1]


def create_test_set_dataframe_from_one_entry(
    data_entry: DataEntry, is_multi_series: bool, offset: int,
) -> pd.DataFrame:
    data_frame = create_dataframe_from_one_data_entry(data_entry, is_multi_series)
    return data_frame[offset+1: ]


def combine_dataframes_of_multiple_series(dataframe_list: List[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(dataframe_list, axis=0, ignore_index=True)


def create_test_set_dataframe(dataset: Dataset) -> pd.DataFrame:
    is_multi_series = is_multi_series_dataset(dataset)
    dataframe_list = [
        create_test_set_dataframe_from_one_entry(data_entry, is_multi_series, dataset.validation_offset_from_the_end)
        for data_entry in dataset.full_dataset
    ]
    return (
        combine_dataframes_of_multiple_series(dataframe_list)
        if is_multi_series
        else dataframe_list[0]
    )


def create_training_set_dataframe(dataset: Dataset) -> pd.DataFrame:
    is_multi_series = is_multi_series_dataset(dataset)
    dataframe_list = [
        create_training_set_dataframe_from_one_entry(data_entry, is_multi_series, dataset.validation_offset_from_the_end)
        for data_entry in dataset.full_dataset
    ]
    return (
        combine_dataframes_of_multiple_series(dataframe_list)
        if is_multi_series
        else dataframe_list[0]
    )


def save_dataframe(dataframe: pd.DataFrame, output_path: Path) -> None:
    dataframe.to_csv(output_path, index=False)


def better_gift_eval_dataset_name(dataset_name: str) -> str:
    return "_".join(dataset_name.split("/"))
