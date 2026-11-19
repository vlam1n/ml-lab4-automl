# общие функции для всех скриптов лабы (данные, разбиение, метрики, запись результатов)
# этот файл импортируется из всех окружений, поэтому тут только pandas/numpy/sklearn

import os
import time
import platform
import resource

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
PREDS_DIR = os.path.join(RESULTS_DIR, "preds")
REPORTS_DIR = os.path.join(RESULTS_DIR, "automl_reports")
for _d in [MODELS_DIR, PREDS_DIR, REPORTS_DIR, os.path.join(RESULTS_DIR, "plots")]:
    os.makedirs(_d, exist_ok=True)

TARGET = "target"
N_CPU = 4            # всем системам даём одинаково 4 потока (ноутбук Apple M3, 8 ядер)
MAIN_SEED = 42       # сид основного разбиения train/test

# в OpenML признаки называются V1..V27, порядок такой же как в оригинале на UCI
FEATURE_NAMES = [
    "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas",
    "X_Perimeter", "Y_Perimeter", "Sum_of_Luminosity", "Minimum_of_Luminosity",
    "Maximum_of_Luminosity", "Length_of_Conveyer", "TypeOfSteel_A300",
    "TypeOfSteel_A400", "Steel_Plate_Thickness", "Edges_Index", "Empty_Index",
    "Square_Index", "Outside_X_Index", "Edges_X_Index", "Edges_Y_Index",
    "Outside_Global_Index", "LogOfAreas", "Log_X_Index", "Log_Y_Index",
    "Orientation_Index", "Luminosity_Index", "SigmoidOfAreas",
]

RESULT_COLUMNS = [
    "system", "budget_min", "seed", "split", "f1_macro_test", "accuracy_test",
    "train_time_s", "predict_time_1000_s", "predict_one_ms", "size_mb",
    "peak_mem_mb", "n_cpu",
]


def load_data():
    return pd.read_csv(os.path.join(DATA_DIR, "steel.csv"))


def get_train_test(split="main"):
    """split='main' - основное разбиение (индексы сохранены в файл один раз и навсегда)
    split=число - другое случайное разбиение 80/20, нужно для гипотезы H4"""
    df = load_data()
    if str(split) == "main":
        train_idx = pd.read_csv(os.path.join(DATA_DIR, "train_idx.csv"))["idx"].values
        test_idx = pd.read_csv(os.path.join(DATA_DIR, "test_idx.csv"))["idx"].values
    else:
        train_idx, test_idx = make_split(df, int(split))
    # важно: loc по перемешанным индексам, т.к. исходный файл отсортирован по классу
    train = df.loc[train_idx].reset_index(drop=True)
    test = df.loc[test_idx].reset_index(drop=True)
    return train, test


def make_split(df, random_state):
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, stratify=df[TARGET], random_state=random_state, shuffle=True
    )
    return train_idx, test_idx


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str).ravel()
    return f1_score(y_true, y_pred, average="macro"), accuracy_score(y_true, y_pred)


def rows_1000(test):
    # в тесте ~390 строк, для замера скорости делаем ровно 1000 строк (с повторами)
    return test.drop(columns=[TARGET]).sample(1000, replace=True, random_state=0).reset_index(drop=True)


def measure_predict_time(predict_fn, test, n_one=20):
    """predict_fn принимает DataFrame без таргета.
    возвращает (время на 1000 строк в сек, медианное время одного предсказания в мс)
    n_one - сколько раз мерить одиночное предсказание (для медленного TabPFN меньше)"""
    X1000 = rows_1000(test)
    predict_fn(X1000.iloc[:10])  # прогрев
    t = time.perf_counter()
    predict_fn(X1000)
    t1000 = time.perf_counter() - t

    one_times = []
    for i in range(n_one):
        row = X1000.iloc[[i]]
        t = time.perf_counter()
        predict_fn(row)
        one_times.append((time.perf_counter() - t) * 1000)
    return t1000, float(np.median(one_times))


def peak_mem_mb():
    # максимум памяти процесса (и дочерних процессов, если были)
    m = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    if platform.system() == "Darwin":
        return m / 1024 / 1024   # на маке в байтах
    return m / 1024              # на линуксе в килобайтах


def size_mb(path):
    if os.path.isfile(path):
        return os.path.getsize(path) / 1024 / 1024
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / 1024 / 1024


def run_name(system, budget, seed, split):
    budget = float(budget)
    b = int(budget) if budget.is_integer() else budget   # 5.0 -> 5, 0.5 -> 0.5
    return f"{system}_b{b}_s{int(seed)}_{split}"


def save_result(row, out="results.csv"):
    path = os.path.join(RESULTS_DIR, out)
    df = pd.DataFrame([row])[RESULT_COLUMNS]
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    print("RESULT:", row)


def save_preds(name, y_true, y_pred):
    os.makedirs(PREDS_DIR, exist_ok=True)
    pd.DataFrame({"y_true": np.asarray(y_true).astype(str),
                  "y_pred": np.asarray(y_pred).astype(str).ravel()}).to_csv(
        os.path.join(PREDS_DIR, name + ".csv"), index=False)
