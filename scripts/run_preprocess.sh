#!/usr/bin/env bash
set -e
python -m citysketch.preprocess --group_dir data/Group1 --out graphs/Group1 --adaptive_k --topo_tol 0.5
python -m citysketch.preprocess --group_dir data/Group2 --out graphs/Group2 --adaptive_k --topo_tol 0.5
python -m citysketch.preprocess_real --real_dir data/RealMap --out graphs/RealMap --adaptive_k --topo_tol 0.5
