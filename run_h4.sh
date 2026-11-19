#!/bin/bash
# гипотеза H4 (малые данные): 5 других случайных разбиений 80/20 (random_state 100..104)
# на каждом: LightAutoML и AutoGluon по 5 минут, TabPFN и CatBoost по умолчанию (B2)
# + к ним в анализе добавляется основное разбиение из results.csv (бюджет 5, сид 0)
# запуск: caffeinate -i bash run_h4.sh   (~1 час). результаты в results/h4.csv

cd "$(dirname "$0")"
eval "$(conda shell.bash hook)"
mkdir -p logs

run() {
    env=$1; name=$2; shift 2
    conda activate "$env"
    echo "$(date '+%H:%M:%S') старт $name"
    python "$@" > "logs/$name.log" 2>&1 || echo "    !!! $name упал, смотри logs/$name.log"
}

for sp in 100 101 102 103 104; do
    run lab4_base   H4_B2_${sp}     run_baselines.py --split $sp --only B2 --seeds 0 --out h4.csv
    run lab4_lama   H4_LAMA_${sp}   run_lama.py      --budget 5 --seed 0 --split $sp --out h4.csv
    run lab4_ag     H4_AG_${sp}     run_autogluon.py --budget 5 --seed 0 --split $sp --out h4.csv
    run lab4_tabpfn H4_TabPFN_${sp} run_tabpfn.py    --seed 0 --split $sp --out h4.csv
done

echo "$(date '+%H:%M:%S') H4 закончена"
