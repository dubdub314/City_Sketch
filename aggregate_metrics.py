import os, pandas as pd

def collect_results(root):
    rows = []
    for exp in os.listdir(root):
        exp_dir = os.path.join(root, exp)
        if not os.path.isdir(exp_dir): continue
        metrics_file = os.path.join(exp_dir, "group_group_metrics.csv")
        if os.path.exists(metrics_file):
            df = pd.read_csv(metrics_file)
            df["exp"] = exp
            rows.append(df)
    if rows:
        return pd.concat(rows)
    return pd.DataFrame()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", required=True)
    args = parser.parse_args()

    df = collect_results(args.results_root)
    out = os.path.join(args.results_root, "summary.csv")
    df.to_csv(out, index=False)
    print(f"Summary saved to {out}")
