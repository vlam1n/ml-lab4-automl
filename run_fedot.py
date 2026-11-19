# FEDOT (ИТМО, лаборатория NSS) - эволюционный поиск пайплайна
# запуск: conda activate lab4_fedot && python run_fedot.py --budget 5 --seed 0
import argparse
import logging
import os
import time

import numpy as np
from fedot.api.main import Fedot
from sklearn.metrics import f1_score

from common import *


# встроенная метрика 'f1' в FEDOT для многоклассовой задачи считается с average='weighted'
# (fedot/core/composer/metrics.py, класс F1), а у нас основная метрика F1-macro.
# поэтому передаём свою функцию. FEDOT метрики минимизирует, поэтому возвращаем минус F1
def f1_macro_metric(pipeline, reference_data, validation_blocks=None):
    predicted = pipeline.predict(reference_data, output_mode="labels")
    return -f1_score(np.ravel(reference_data.target), np.ravel(predicted.predict), average="macro")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, required=True, help="минуты")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="main")
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()

    train, test = get_train_test(args.split)
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET].values
    y_test = test[TARGET]

    model = Fedot(
        problem="classification",
        timeout=args.budget,          # у FEDOT таймаут в минутах
        metric=f1_macro_metric,
        seed=args.seed,
        n_jobs=N_CPU,
        logging_level=logging.ERROR,
    )

    t = time.perf_counter()
    model.fit(features=X_train, target=y_train)
    train_time = time.perf_counter() - t

    def predict_labels(X):
        return np.ravel(model.predict(features=X))

    y_pred = predict_labels(test.drop(columns=[TARGET]))
    f1, acc = calc_metrics(y_test, y_pred)
    t1000, t_one = measure_predict_time(predict_labels, test)

    rn = run_name("FEDOT", args.budget, args.seed, args.split)
    path = os.path.join(MODELS_DIR, rn)
    model.current_pipeline.save(path=path, create_subdir=False, is_datetime_in_path=False)

    # что выбрал AutoML: структура пайплайна (граф) и картинка
    with open(os.path.join(REPORTS_DIR, rn + "_structure.txt"), "w") as f:
        f.write(str(model.current_pipeline) + "\n\n")
        for node in model.current_pipeline.nodes:
            f.write(f"{node.name}: parents={[p.name for p in node.nodes_from]} params={node.parameters}\n")
    try:
        model.current_pipeline.show(save_path=os.path.join(REPORTS_DIR, rn + "_pipeline.png"))
    except Exception as e:
        print("не получилось нарисовать пайплайн:", e)

    save_preds(rn, y_test, y_pred)
    save_result({
        "system": "FEDOT", "budget_min": args.budget, "seed": args.seed, "split": args.split,
        "f1_macro_test": f1, "accuracy_test": acc, "train_time_s": train_time,
        "predict_time_1000_s": t1000, "predict_one_ms": t_one,
        "size_mb": size_mb(path), "peak_mem_mb": peak_mem_mb(), "n_cpu": N_CPU,
    }, args.out)
