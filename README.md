# CityCog-GNN Ablation Suite

This package controls ablation experiments for the CityCog-GNN project.

## Modules toggled
- Multi-scale ring signatures (`CITYSKETCH_USE_RING=1/0`)
- Soft-topology weighting (`CITYSKETCH_SOFT_TOPO=1/0`)
- Hierarchical staging (`CITYSKETCH_STAGE_HIER=1/0`)

## Usage
1. Unzip your main CityCog-GNN project somewhere.
2. Set environment variable:
   export MAIN_PROJ_ROOT=/path/to/citysketch_full_project
3. Run:
   python run_ablation.py --dataset_root /path/to/datasets --out_dir ./results
