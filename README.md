# ЛР 4. AutoML в инженерных задачах — вариант 3 (Steel Plates Faults)

Классификация 7 типов дефектов стальных листов (OpenML 40982, 1941 строка, 27 признаков).
Сравниваем ручные решения (B0–B3) с LightAutoML, FEDOT (российские) и AutoGluon, плюс TabPFN.
Гипотезы: H1, H2 (обязательные) и H4 (малые данные).

## Структура
```
common.py            общие функции: данные, разбиение, метрики, замеры, запись results.csv
prepare_data.py      arff -> data/steel.csv + один раз откладывает тест (data/test_idx.csv)
run_baselines.py     B0 (константа), B1 (логрег в Pipeline), B2 (CatBoost по умолчанию)
run_b3_optuna.py     B3: CatBoost + Optuna с тем же бюджетом времени
run_lama.py          LightAutoML
run_fedot.py         FEDOT
run_autogluon.py     AutoGluon
run_tabpfn.py        TabPFN (бонус + H4)
run_main.sh          весь основной эксперимент (бюджеты 1/5/15 мин x сиды 0/1/2)
run_h4.sh            гипотеза H4 (5 дополнительных разбиений)
setup_envs.sh        создание 5 conda-окружений
01_eda.ipynb         EDA
02_analysis.ipynb    таблицы, графики, проверка гипотез
hypotheses.md        прогноз по гипотезам (от 18.11.2026, до запуска экспериментов)
report.md            отчёт: результаты, проверка гипотез, ограничения, служебная записка заказчику
Теория_к_защите_ЛР4.docx  теория для подготовки к защите
data/                датасет и индексы разбиения
envs/                точные версии пакетов каждого окружения
results/             results.csv, h4.csv, preds/, plots/, automl_reports/
models/              сохранённые модели (для замера размера)
logs/                логи каждого запуска
```

## Окружения
У библиотек конфликтующие зависимости, поэтому 5 отдельных conda-окружений (Python 3.11):

| окружение | что внутри | зачем |
|---|---|---|
| lab4_base | sklearn, catboost, optuna, jupyter | B0–B3, EDA, анализ |
| lab4_lama | lightautoml 0.4.2 | LightAutoML |
| lab4_fedot | fedot 0.7.5 (+ pandas<2.2, numpy<2, setuptools<81) | FEDOT |
| lab4_ag | autogluon.tabular[all] 1.6.3 | AutoGluon |
| lab4_tabpfn | tabpfn 2.0.9 | TabPFN |

`bash setup_envs.sh` — создаёт всё по `envs/requirements_*.txt`.

Грабли, на которые наступили (пригодятся на защите и в «Ограничениях»):
1. **macOS + две копии OpenMP.** torch приносит свою `libomp.dylib`, LightGBM берёт libomp из окружения.
   LightAutoML зависал на LightGBM с 0% CPU. Лечение — в `setup_envs.sh` (torch смотрит на общую libomp).
2. **FEDOT 0.7.5** не работает с pandas 3 (`applymap`) и новым setuptools (`pkg_resources`) — версии закреплены.
3. **FEDOT считает `'f1'` для многих классов как weighted F1**, а не macro — передаём свою метрику.
4. **TabPFN** новых версий требует регистрацию и ключ Prior Labs, взяли 2.0.9 (веса с HuggingFace).
   На CPU он медленный (~40 с на любой вызов predict), MPS (GPU мака) в 2.0.9 не поддерживается.
5. **Данные отсортированы по классу** — всё разбиение только с перемешиванием.

## Порядок запуска
```bash
conda activate lab4_base
python prepare_data.py                 # один раз (тест уже отложен, повторно не перезапишет)
jupyter notebook 01_eda.ipynb          # EDA

# 1) заполнить hypotheses.md и закоммитить ДО экспериментов
git add hypotheses.md && git commit -m "прогноз по гипотезам до эксперимента"

# 2) эксперименты (на ночь, ноутбук на зарядке, ничего тяжёлого параллельно не запускать)
caffeinate -i bash run_main.sh         # ~4.5-5 ч
caffeinate -i bash run_h4.sh           # ~1-1.5 ч

# 3) анализ
jupyter notebook 02_analysis.ipynb
```
Один запуск вручную, например: `conda activate lab4_lama && python run_lama.py --budget 5 --seed 0`.
Если какой-то запуск упал — смотреть `logs/<имя>.log`, перезапустить только его (строка допишется в results.csv;
старую строку упавшего запуска, если она есть, удалить руками).

## Протокол
- Тест: 20%, стратификация, `random_state=42`, `data/test_idx.csv`. Все системы учатся только на 80%.
- Метрика: F1-macro (основная), accuracy, матрица ошибок.
- Железо: Apple M3, 8 ядер, 16 ГБ; каждой системе `N_CPU = 4` потока (`common.py`).
- В results.csv на каждый запуск: система, бюджет, сид, F1/accuracy на тесте, фактическое время обучения,
  время предсказания 1000 строк, медианное время одного предсказания, размер модели на диске, пиковая память.
- Пиковая память — `ru_maxrss` процесса (и завершившихся дочерних), это приближение.
