#!/bin/bash
# создание окружений: каждая AutoML-библиотека в своём conda-окружении (иначе конфликты версий)
# точные версии всех пакетов лежат в envs/requirements_*.txt (pip list --format=freeze)
# запуск: bash setup_envs.sh   (качает несколько ГБ, ~15-20 минут)

cd "$(dirname "$0")"
eval "$(conda shell.bash hook)"

for e in base lama fedot ag tabpfn; do
    # llvm-openmp из conda-forge - общая libomp для lightgbm/catboost/torch
    conda create -y -n lab4_$e -c conda-forge python=3.11 llvm-openmp
    conda activate lab4_$e
    pip install -r envs/requirements_$e.txt
    conda deactivate
done

# ВАЖНО для macOS: torch тащит свою копию libomp.dylib, а lightgbm берёт libomp из окружения.
# две копии OpenMP в одном процессе -> LightAutoML намертво зависал на LightGBM (0% CPU),
# а при другом порядке импорта был segfault. лечится тем, что torch смотрит на ту же libomp
for e in lama ag tabpfn; do
    E=$(conda info --base)/envs/lab4_$e
    T=$E/lib/python3.11/site-packages/torch/lib
    if [ -f "$T/libomp.dylib" ] && [ ! -L "$T/libomp.dylib" ]; then
        mv "$T/libomp.dylib" "$T/libomp.dylib.orig"
        ln -s "$E/lib/libomp.dylib" "$T/libomp.dylib"
        echo "libomp исправлена в lab4_$e"
    fi
done
