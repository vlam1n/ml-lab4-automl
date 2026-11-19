# B0, B1, B2 - классические решения без подбора гиперпараметров
# запуск: conda activate lab4_base && python run_baselines.py --split main
import argparse
import os
import time

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import *

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="main")
parser.add_argument("--out", default="results.csv")
parser.add_argument("--seeds", default="0,1,2")
parser.add_argument("--only", default="B0,B1,B2")
args = parser.parse_args()

train, test = get_train_test(args.split)
X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]


def make_model(name, seed):
    if name == "B0":
        return DummyClassifier(strategy="most_frequent")
    if name == "B1":
        # пропусков в данных нет и категориальных признаков нет (TypeOfSteel уже 0/1),
        # но imputer оставляем как требует задание, one-hot тут кодировать нечего
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=5000, random_state=seed)),
        ])
    if name == "B2":
        return CatBoostClassifier(random_seed=seed, thread_count=N_CPU, verbose=0)


for name in args.only.split(","):
    for seed in [int(s) for s in args.seeds.split(",")]:
        model = make_model(name, seed)
        t = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t

        y_pred = model.predict(X_test)
        f1, acc = calc_metrics(y_test, y_pred)
        t1000, t_one = measure_predict_time(model.predict, test)

        rn = run_name(name, 0, seed, args.split)
        if name == "B2":
            path = os.path.join(MODELS_DIR, rn + ".cbm")
            model.save_model(path)
            if seed == 0 and args.split == "main":
                imp = pd.Series(model.get_feature_importance(), index=X_train.columns)
                imp.sort_values(ascending=False).to_csv(os.path.join(REPORTS_DIR, "catboost_B2_importance.csv"))
        else:
            path = os.path.join(MODELS_DIR, rn + ".pkl")
            joblib.dump(model, path)

        save_preds(rn, y_test, y_pred)
        save_result({
            "system": name, "budget_min": 0, "seed": seed, "split": args.split,
            "f1_macro_test": f1, "accuracy_test": acc, "train_time_s": train_time,
            "predict_time_1000_s": t1000, "predict_one_ms": t_one,
            "size_mb": size_mb(path), "peak_mem_mb": peak_mem_mb(), "n_cpu": N_CPU,
        }, args.out)
