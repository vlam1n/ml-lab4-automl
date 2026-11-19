# TabPFN - предобученная модель для маленьких таблиц, без обучения и подбора гиперпараметров
# (бонус + часть гипотезы H4). бюджета времени у неё нет, пишем budget_min = 0
# запуск: conda activate lab4_tabpfn && python run_tabpfn.py --seed 0
# версия tabpfn==2.0.9: веса качаются с HuggingFace без регистрации (новые версии требуют ключ Prior Labs)
import argparse
import os
import time

# на CPU больше 1000 строк TabPFN по умолчанию не пускает (просто потому что медленно), разрешаем
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

import joblib
from tabpfn import TabPFNClassifier

from common import *

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--split", default="main")
parser.add_argument("--out", default="results.csv")
args = parser.parse_args()

train, test = get_train_test(args.split)
X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
y_test = test[TARGET]

model = TabPFNClassifier(device="cpu", random_state=args.seed)
t = time.perf_counter()
model.fit(X_train, y_train)   # тут по сути только запоминаются данные
train_time = time.perf_counter() - t


def predict_labels(X):
    return model.predict(X)


y_pred = predict_labels(test.drop(columns=[TARGET]))
f1, acc = calc_metrics(y_test, y_pred)
# на CPU одно предсказание ~40 с (каждый раз заново прогоняется вся обучающая выборка),
# поэтому одиночное предсказание меряем 3 раза, а не 20
t1000, t_one = measure_predict_time(predict_labels, test, n_one=3)

rn = run_name("TabPFN", 0, args.seed, args.split)
path = os.path.join(MODELS_DIR, rn + ".pkl")
joblib.dump(model, path)   # внутри и веса сети, и обучающая выборка

save_preds(rn, y_test, y_pred)
save_result({
    "system": "TabPFN", "budget_min": 0, "seed": args.seed, "split": args.split,
    "f1_macro_test": f1, "accuracy_test": acc, "train_time_s": train_time,
    "predict_time_1000_s": t1000, "predict_one_ms": t_one,
    "size_mb": size_mb(path), "peak_mem_mb": peak_mem_mb(), "n_cpu": N_CPU,
}, args.out)
