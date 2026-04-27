"""
Dense GCNII Experiment — Table 8 upper bound
Run this on Google Colab where the Unifews environment is set up.

Usage:
  1. Upload to Colab or paste into a cell
  2. Ensure Unifews is at /content/Unifews with deps installed
  3. Run: !python /content/Unifews/run_dense_gcnii_table8.py
"""
import subprocess
import sys
import re
import numpy as np
from collections import defaultdict

SEEDS = [11, 22, 33, 42, 55, 66, 77, 88, 99, 100]
DATASETS = ["cora", "chameleon"]
LAYERS = [2, 4, 8, 16, 32, 64]

results = defaultdict(lambda: defaultdict(list))

for data in DATASETS:
    for L in LAYERS:
        for seed in SEEDS:
            cmd = [
                sys.executable, "run_fb.py",
                "-c", data,
                "-m", "gcn2_thr",
                "-l", str(L),
                "-f", str(seed),
                "-a", "0.0",
                "-w", "0.0",
                "--fa_alpha", "1.0",
            ]
            print(f"  Running {data} L={L} seed={seed}...", end="", flush=True)
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300,
                    cwd="/content/Unifews"
                )
                output = result.stdout + result.stderr
                # Parse test accuracy from output: "[Test] best acc: X.XXXXX"
                match = re.search(r"\[Test\] best acc:\s*([\d.]+)", output)
                if match:
                    acc = float(match.group(1)) * 100  # convert to %
                    results[data][L].append(acc)
                    print(f" acc={acc:.2f}%")
                else:
                    print(f" PARSE FAILED")
                    if result.stderr:
                        print(f"    stderr: {result.stderr[:200]}")
            except subprocess.TimeoutExpired:
                print(f" TIMEOUT")
            except Exception as e:
                print(f" ERROR: {e}")

# ============================================================
# Print summary table (copy these numbers into Table 8)
# ============================================================
print("\n" + "=" * 70)
print("RESULTS: Dense GCNII (no pruning) — for Table 8")
print("=" * 70)

header = f"{'Dataset':<12}" + "".join(f"{'L='+str(L):>10}" for L in LAYERS)
print(header)
print("-" * len(header))

for data in DATASETS:
    row = f"{data:<12}"
    for L in LAYERS:
        accs = results[data][L]
        if accs:
            mean = np.mean(accs)
            std = np.std(accs)
            row += f"{mean:>7.2f}±{std:.1f}"
        else:
            row += f"{'N/A':>10}"
    print(row)

print("\n" + "=" * 70)
print("LaTeX format for Table 8 (Dense GCNII row):")
print("=" * 70)
for data in DATASETS:
    row_parts = [f"Dense GCNII"]
    for L in LAYERS:
        accs = results[data][L]
        if accs:
            mean = np.mean(accs)
            row_parts.append(f"{mean:.2f}")
        else:
            row_parts.append("---")
    print(f"% {data}")
    print("& " + " & ".join(row_parts) + " \\\\")
