# CitySketch (Full Project): Structure-First GNN for Hand-Drawn City Maps

This project contains a **complete** pipeline:

- **Data ingestion** from shapefiles & relations
- **Structure Skeleton Graph (SSG)** construction with rotation/scale invariance
- **Topology & shape descriptors** (points-in-district, on-line, endpoints, circularity/rectangularity/straightness)
- **Unsupervised training** (contrastive; optional FGW regularization)
- **Supervised / semi-supervised training** (pairwise labels, optional SupCon, optional node-level alignment)
- **Group-vs-Group similarity** (m×n matrix + Hungarian pairing + Fidelity)
- **Accuracy vs RealMap** (structure/type/topology/shape fidelity; RealMap shapefile preferred; Google API stub fallback)
- **Score calibration** (Platt / Isotonic) for Fidelity in [0,1]

> Core idea: **Structure-first** — insensitive to absolute coordinates/scale/rotation; sensitive to **relative order, directions, and topology**.

## Install
```bash
pip install -r requirements.txt
# For torch-geometric, follow their platform-specific instructions if pip fails:
# https://pytorch-geometric.readthedocs.io/
```

## Data layout
```
data/
  Group1/
    Sample 001/
      Points.shp ...
      Lines.shp ...
      Polygons.shp ...
      relations.xlsx  (optional; columns: src_name,dst_name,rel_type)
    Sample 002/ ...
  Group2/ ...
  RealMap/          (optional; preferred for accuracy vs real)
    RealPoints.shp ...
    RealLines.shp ...
    RealPolygons.shp ...
```

Shapefile attributes required: `Type` in {Landmark,Node,Path,Edge,District} and `Name` (string). Field names are case-insensitive.

## 1) Preprocess
```bash
# Groups (uses relations by default)
python -m citysketch.preprocess --group_dir data/Group1 --out graphs/Group1
python -m citysketch.preprocess --group_dir data/Group2 --out graphs/Group2

# Real map (no relations)
python -m citysketch.preprocess_real --real_dir data/RealMap --out graphs/RealMap
```

## 2) Train

### A) Unsupervised (self-supervised contrastive)
```bash
python -m citysketch.train --graphs_root graphs --epochs 50 --batch_size 8   --fgw_weight 0.0   # set >0 only if you have POT installed and want FGW regularization
```
Saves `graphs/ckpt_structure_first.pt`.

### B) Supervised / semi-supervised (pairwise labels; optional SupCon)
Prepare `data/labels/pairs.csv`:
```csv
sample_a,sample_b,label,weight
Sample 001,Sample 104,1,1.0
Sample 002,Sample 103,0,1.0
Sample 003,Sample 105,0.85,0.5
```
Train:
```bash
python -m citysketch.train_sup   --graphs_root graphs   --pairs_csv data/labels/pairs.csv   --losses "{"w_contrast":0.5,"w_pair":1.0,"w_supcon":0.0}"   --pair_loss bce   --epochs 50 --batch_size 8
```
Saves `graphs/ckpt_sup.pt`.

## 3) Evaluate

### Group vs Group (similarity)
```bash
python -m citysketch.evaluate_groups   --graphs_root graphs   --group_a Group1 --group_b Group2   --out reports_groups
```
Generates `similarity_matrix.csv` + `report.json` (pairing + group fidelity).  
If both `ckpt_sup.pt` and `ckpt_structure_first.pt` exist, the script prefers **supervised** checkpoint.

### Group vs RealMap (accuracy)
```bash
python -m citysketch.evaluate_accuracy   --graphs_root graphs   --group Group1 --real RealMap   --out reports_accuracy
```
Outputs per-sample Fidelity (0–1) and components (structure/type/topology/shape).

## 4) (Optional) Score Calibration
```bash
python -m citysketch.calibrate   --graphs_root graphs   --pairs_csv data/labels/val_pairs.csv   --ckpt graphs/ckpt_sup.pt   --out calibrator.json
```

## Notes
- Relations (Gaze/Connection/Navigation/Fusion/Abruptness/Penetration) are **used only for group-vs-group** similarity as **low-weight** features.
- Accuracy vs RealMap **does not use relations**; it relies on structure/type/topology/shape.
- Topology checks include point-in-district, on-line (with buffer tolerance), near-line-endpoint, touches-boundary (tolerance configurable).
