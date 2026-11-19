# шаг 0: переводим arff с OpenML (id 40982) в csv и ОДИН РАЗ откладываем тестовую выборку
# запуск: conda activate lab4_base && python prepare_data.py
import os
import pandas as pd
from scipy.io import arff

from common import DATA_DIR, TARGET, FEATURE_NAMES, MAIN_SEED, make_split

raw, meta = arff.loadarff(os.path.join(DATA_DIR, "steel-plates-fault.arff"))
df = pd.DataFrame(raw)
df[TARGET] = df[TARGET].str.decode("utf-8")
df.columns = FEATURE_NAMES + [TARGET]
df.to_csv(os.path.join(DATA_DIR, "steel.csv"), index=False)
print(df.shape)
print(df[TARGET].value_counts())

test_path = os.path.join(DATA_DIR, "test_idx.csv")
if os.path.exists(test_path):
    print("test_idx.csv уже есть, не перезаписываю (тест откладывается один раз)")
else:
    train_idx, test_idx = make_split(df, MAIN_SEED)  # 20%, стратификация, перемешивание
    pd.DataFrame({"idx": train_idx}).to_csv(os.path.join(DATA_DIR, "train_idx.csv"), index=False)
    pd.DataFrame({"idx": test_idx}).to_csv(test_path, index=False)
    print("train:", len(train_idx), "test:", len(test_idx))
