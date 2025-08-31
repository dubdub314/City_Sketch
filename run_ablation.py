import os, subprocess, yaml, argparse

def run_exp(exp, args):
    env = os.environ.copy()
    env["CITYSKETCH_USE_RING"] = str(exp["ring"])
    env["CITYSKETCH_SOFT_TOPO"] = str(exp["soft_topo"])
    env["CITYSKETCH_STAGE_HIER"] = str(exp["hier"])
    env["CITYSKETCH_HYPER"] = "1"

    exp_dir = os.path.join(args.out_dir, exp["name"])
    os.makedirs(exp_dir, exist_ok=True)

    steps = [
        f"bash {args.main_root}/scripts/run_preprocess.sh {args.dataset_root} {exp_dir}",
        f"bash {args.main_root}/scripts/train_unsup.sh {exp_dir}",
        f"bash {args.main_root}/scripts/train_sup.sh {exp_dir}",
        f"bash {args.main_root}/scripts/eval_groups.sh {exp_dir}",
        f"bash {args.main_root}/scripts/eval_real.sh {exp_dir}"
    ]
    for cmd in steps:
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, env=env, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--main_root", default=os.environ.get("MAIN_PROJ_ROOT"))
    args = parser.parse_args()

    with open("ablation_matrix.yaml") as f:
        matrix = yaml.safe_load(f)["experiments"]

    for exp in matrix:
        print(f"=== Experiment {exp['name']} ===")
        run_exp(exp, args)

if __name__ == "__main__":
    main()
