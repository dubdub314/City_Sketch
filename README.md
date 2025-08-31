# CitySketch (Full Project): Structure-First GNN for Hand-Drawn City Maps

pipeline:

- **Data ingestion** from shapefiles & relations
- **Structure Skeleton Graph (SSG)** construction with rotation/scale invariance
- **Topology & shape descriptors** (points-in-district, on-line, endpoints, circularity/rectangularity/straightness)
- **Unsupervised training** (contrastive; optional FGW regularization)
- **Supervised / semi-supervised training** (pairwise labels, optional SupCon, optional node-level alignment)
- **Group-vs-Group similarity** (m×n matrix + Hungarian pairing + Fidelity)
- **Accuracy vs RealMap** (structure/type/topology/shape fidelity; RealMap shapefile preferred; Google API stub fallback)
- **Score calibration** (Platt / Isotonic) for Fidelity in [0,1]

> Core idea: **Structure-first** — insensitive to absolute coordinates/scale/rotation; sensitive to **relative order, directions, and topology**.


