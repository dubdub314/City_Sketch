#!/usr/bin/env bash
set -e
python -m citysketch.evaluate_groups --graphs_root graphs --group_a Group1 --group_b Group2 --out reports/groups
python -m citysketch.evaluate_accuracy --graphs_root graphs --group Group1 --real RealMap --out reports/acc
