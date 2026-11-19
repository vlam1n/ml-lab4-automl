# LightAutoML (Sber AI Lab)
# запуск: conda activate lab4_lama && python run_lama.py --budget 5 --seed 0
import argparse
import os
import time

import joblib
import numpy as np
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

from common import *

parser = argparse.ArgumentParser()
parser.add_argument("--budget", type=float, required=True, help="минуты")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--split", default="main")
parser.add_argument("--out", default="results.csv")
args = parser.parse_args()

train, test = get_train_test(args.split)
y_test = test[TARGET]

# метрику задаём явно, иначе LAMA выбирает лучшую модель по crossentropy, а нам нужен F1-macro
task = Task("multiclass", metric="f1_macro")
automl = TabularAutoML(
    task=task,
    timeout=int(args.budget * 60),
    cpu_limit=N_CPU,
    reader_params={"random_state": args.seed, "n_jobs": N_CPU, "cv": 5},
)

t = time.perf_counter()
automl.fit_predict(train, roles={"target": TARGET}, verbose=1)
train_time = time.perf_counter() - t

# LAMA возвращает вероятности, классы закодированы числами -> переводим обратно в названия
mapping = automl.reader.class_mapping  # {'Bumps': 0, ...}
inv = {v: k for k, v in mapping.items()} if mapping else None


def predict_labels(X):
    proba = automl.predict(X).data
    idx = np.argmax(proba, axis=1)
    return np.array([inv[i] for i in idx]) if inv else idx


y_pred = predict_labels(test.drop(columns=[TARGET]))
f1, acc = calc_metrics(y_test, y_pred)
t1000, t_one = measure_predict_time(predict_labels, test)

rn = run_name("LightAutoML", args.budget, args.seed, args.split)
path = os.path.join(MODELS_DIR, rn + ".pkl")
joblib.dump(automl, path)

# что выбрал AutoML: состав модели и важность признаков
with open(os.path.join(REPORTS_DIR, rn + "_structure.txt"), "w") as f:
    f.write(automl.create_model_str_desc())
if args.split == "main" and args.seed == 0:
    automl.get_feature_scores("fast").to_csv(os.path.join(REPORTS_DIR, rn + "_importance.csv"), index=False)

save_preds(rn, y_test, y_pred)
save_result({
    "system": "LightAutoML", "budget_min": args.budget, "seed": args.seed, "split": args.split,
    "f1_macro_test": f1, "accuracy_test": acc, "train_time_s": train_time,
    "predict_time_1000_s": t1000, "predict_one_ms": t_one,
    "size_mb": size_mb(path), "peak_mem_mb": peak_mem_mb(), "n_cpu": N_CPU,
}, args.out)
