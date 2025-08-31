#!/usr/bin/env bash
set -e
python -m citysketch.train --graphs_root graphs --groups Group1,Group2 --epochs 50 --batch_size 8 --log_dir runs/pretrain
