#!/bin/bash
# основной эксперимент: B0-B2, потом B3 (Optuna) и три AutoML с бюджетами 1, 5, 15 минут и сидами 0, 1, 2
# все запуски строго по очереди на одном ноутбуке (Apple M3, 8 ядер, 16 ГБ), каждой системе 4 потока
# запуск из папки лабы:  caffeinate -i bash run_main.sh     (caffeinate - чтобы мак не уснул)
# идёт примерно 4.5-5 часов. результаты дописываются в results/results.csv

cd "$(dirname "$0")"
eval "$(conda shell.bash hook)"
mkdir -p logs

run() {
    env=$1; name=$2; shift 2
    conda activate "$env"
    echo "$(date '+%H:%M:%S') старт $name"
    python "$@" > "logs/$name.log" 2>&1 || echo "    !!! $name упал, смотри logs/$name.log"
}

run lab4_base baselines run_baselines.py --split main

for b in 1 5 15; do
    for s in 0 1 2; do
        run lab4_base  B3_b${b}_s${s}    run_b3_optuna.py --budget $b --seed $s
        run lab4_lama  LAMA_b${b}_s${s}  run_lama.py      --budget $b --seed $s
        run lab4_fedot FEDOT_b${b}_s${s} run_fedot.py     --budget $b --seed $s
        run lab4_ag    AG_b${b}_s${s}    run_autogluon.py --budget $b --seed $s
    done
done

# бонус: TabPFN на основном разбиении
for s in 0 1 2; do
    run lab4_tabpfn TabPFN_s${s} run_tabpfn.py --seed $s
done

echo "$(date '+%H:%M:%S') основной эксперимент закончен"
