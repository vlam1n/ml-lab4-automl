# AutoGluon (Amazon) - многослойный стекинг
# запуск: conda activate lab4_ag && python run_autogluon.py --budget 5 --seed 0
import argparse
import os
import shutil
import time

from autogluon.tabular import TabularPredictor

from common import *

parser = argparse.ArgumentParser()
parser.add_argument("--budget", type=float, required=True, help="минуты")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--split", default="main")
parser.add_argument("--out", default="results.csv")
args = parser.parse_args()

train, test = get_train_test(args.split)
y_test = test[TARGET]

rn = run_name("AutoGluon", args.budget, args.seed, args.split)
path = os.path.join(MODELS_DIR, rn)
if os.path.exists(path):
    shutil.rmtree(path)

t = time.perf_counter()
# random_state у learner отвечает за разбиение train/holdout внутри AutoGluon - через него меняем сид
predictor = TabularPredictor(
    label=TARGET, eval_metric="f1_macro", path=path, verbosity=1,
    learner_kwargs={"random_state": args.seed},
).fit(train, time_limit=int(args.budget * 60), presets="medium_quality", num_cpus=N_CPU)
train_time = time.perf_counter() - t


def predict_labels(X):
    return predictor.predict(X).values


y_pred = predict_labels(test.drop(columns=[TARGET]))
f1, acc = calc_metrics(y_test, y_pred)
t1000, t_one = measure_predict_time(predict_labels, test)

# что выбрал AutoML: лидерборд моделей (оценки на внутренней валидации, не на тесте)
predictor.leaderboard(silent=True).to_csv(os.path.join(REPORTS_DIR, rn + "_leaderboard.csv"), index=False)
with open(os.path.join(REPORTS_DIR, rn + "_structure.txt"), "w") as f:
    f.write(f"лучшая модель: {predictor.model_best}\n")
    try:
        best = predictor._trainer.load_model(predictor.model_best)
        f.write(f"веса ансамбля: {best._get_model_weights()}\n")
    except Exception as e:
        f.write(f"веса не достались ({e})\n")

save_preds(rn, y_test, y_pred)
peak = peak_mem_mb()
save_result({
    "system": "AutoGluon", "budget_min": args.budget, "seed": args.seed, "split": args.split,
    "f1_macro_test": f1, "accuracy_test": acc, "train_time_s": train_time,
    "predict_time_1000_s": t1000, "predict_one_ms": t_one,
    "size_mb": size_mb(path), "peak_mem_mb": peak, "n_cpu": N_CPU,
}, args.out)

# важность признаков считаем один раз (перестановочная важность на отложенной выборке,
# только для интерпретации - ничего по ней не выбираем)
if args.split == "main" and args.seed == 0 and args.budget == 5:
    fi = predictor.feature_importance(test, time_limit=120)
    fi.to_csv(os.path.join(REPORTS_DIR, rn + "_importance.csv"))
