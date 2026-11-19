# B3 - тот же CatBoost, но гиперпараметры подбираем Optuna с таким же бюджетом времени как у AutoML
# качество на подборе считаем кросс-валидацией ТОЛЬКО на train, тест не трогаем
# запуск: conda activate lab4_base && python run_b3_optuna.py --budget 5 --seed 0
import argparse
import os
import time

import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from common import *

parser = argparse.ArgumentParser()
parser.add_argument("--budget", type=float, required=True, help="минуты")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--split", default="main")
parser.add_argument("--out", default="results.csv")
args = parser.parse_args()

train, test = get_train_test(args.split)
X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)


def objective(trial):
    params = {
        "depth": trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "iterations": trial.suggest_int("iterations", 200, 1000, step=100),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
        # для F1-macro важны редкие классы, поэтому пробуем и веса классов
        "auto_class_weights": trial.suggest_categorical("auto_class_weights", ["None", "Balanced"]),
    }
    if params["auto_class_weights"] == "None":
        params["auto_class_weights"] = None
    model = CatBoostClassifier(**params, random_seed=args.seed, thread_count=N_CPU, verbose=0)
    return cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro").mean()


optuna.logging.set_verbosity(optuna.logging.WARNING)
t = time.perf_counter()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
study.optimize(objective, timeout=args.budget * 60)

best = dict(study.best_params)
if best["auto_class_weights"] == "None":
    best["auto_class_weights"] = None
model = CatBoostClassifier(**best, random_seed=args.seed, thread_count=N_CPU, verbose=0)
model.fit(X_train, y_train)
train_time = time.perf_counter() - t  # время подбора + финальное обучение
print("trials:", len(study.trials), "best cv f1:", study.best_value, best)

y_pred = model.predict(X_test)
f1, acc = calc_metrics(y_test, y_pred)
t1000, t_one = measure_predict_time(model.predict, test)

rn = run_name("B3_optuna", args.budget, args.seed, args.split)
path = os.path.join(MODELS_DIR, rn + ".cbm")
model.save_model(path)
with open(os.path.join(REPORTS_DIR, rn + ".txt"), "w") as f:
    f.write(f"trials: {len(study.trials)}\nbest cv f1_macro: {study.best_value}\nparams: {best}\n")

save_preds(rn, y_test, y_pred)
save_result({
    "system": "B3_optuna", "budget_min": args.budget, "seed": args.seed, "split": args.split,
    "f1_macro_test": f1, "accuracy_test": acc, "train_time_s": train_time,
    "predict_time_1000_s": t1000, "predict_one_ms": t_one,
    "size_mb": size_mb(path), "peak_mem_mb": peak_mem_mb(), "n_cpu": N_CPU,
}, args.out)
