#!/usr/bin/env bash
set -e
python -m citysketch.train_sup --graphs_root graphs --pairs_csv data/pairs.csv --epochs 50 --pair_loss bce --log_dir runs/sup
