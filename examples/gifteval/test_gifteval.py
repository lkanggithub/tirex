from tirex import ForecastModel, load_model

model: ForecastModel = load_model("NX-AI/TiRex", device="cpu")


import pandas as pd
from gift_eval_utils import TiRexGiftEvalWrapper, evaluate_dataset, gift_eval_dataset_iter

wrapped_model = TiRexGiftEvalWrapper(model)
results = []
for task in gift_eval_dataset_iter():
    task_result = evaluate_dataset(wrapped_model, **task)
    results.append(task_result)
    print(task_result)
results = pd.DataFrame(results)
print(results)
