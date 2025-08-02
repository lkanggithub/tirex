import logging
from dataclasses import dataclass
from typing import Any

from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality

from data import TestDataset


# avoid excessive logging
class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record):
        return self.text_to_filter not in record.getMessage()


gts_logger = logging.getLogger("gluonts.model.forecast")
gts_logger.addFilter(WarningFilter("The mean prediction is not stored in the forecast data"))

METRICS = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
]


# Setup GiftEval evaluation
def evaluate_dataset(predictor, dataset: TestDataset):
    predictor.set_prediction_len(dataset.prediction_length)
    predictor.set_ds_freq(dataset.time_series_frequency)
    predictor.set_context_len(dataset.prediction_length)  # FIXME
    season_length = get_seasonality(dataset.time_series_frequency)

    # Measure the time taken for evaluation
    res = evaluate_model(
        predictor,
        test_data=dataset.test_data,
        metrics=METRICS,
        batch_size=1024,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season_length,
    )
    result = {
        "dataset": dataset.name,
        "model": predictor.model_id,
        "eval_metrics/MSE[mean]": res["MSE[mean]"][0],
        "eval_metrics/MSE[0.5]": res["MSE[0.5]"][0],
        "eval_metrics/MAE[0.5]": res["MAE[0.5]"][0],
        "eval_metrics/MASE[0.5]": res["MASE[0.5]"][0],
        "eval_metrics/MAPE[0.5]": res["MAPE[0.5]"][0],
        "eval_metrics/sMAPE[0.5]": res["sMAPE[0.5]"][0],
        "eval_metrics/MSIS": res["MSIS"][0],
        "eval_metrics/RMSE[mean]": res["RMSE[mean]"][0],
        "eval_metrics/NRMSE[mean]": res["NRMSE[mean]"][0],
        "eval_metrics/ND[0.5]": res["ND[0.5]"][0],
        "eval_metrics/mean_weighted_sum_quantile_loss": res["mean_weighted_sum_quantile_loss"][0],
    }
    return result


@dataclass
class TiRexGiftEvalWrapper:
    model: Any
    freq: str = None
    pred_len: int = 32
    context_len: int = 128

    def set_ds_freq(self, freq):
        self.freq = freq

    def set_prediction_len(self, pred_len):
        self.pred_len = pred_len

    def set_context_len(self, context_len: int) -> None:
        self.context_len = context_len

    def predict(self, test_data_input):
        forecasts = self.model.forecast_gluon(
            test_data_input,
            prediction_length=self.pred_len,
            output_type="gluonts",
            predict_context_length=self.context_len,
        )
        return forecasts

    @property
    def model_id(self):
        return "TiRex"
