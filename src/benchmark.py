import os
import re
import sys
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = os.path.join(HERE, "recommender_seq.py")
MPI = os.path.join(HERE, "recommender_mpi.py")
ROOT = os.path.dirname(HERE)


def run(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        raise RuntimeError(f"Команда завершилась с кодом {proc.returncode}: {' '.join(cmd)}")
    return proc.stdout


def measure_seq(repeats):
    best = None
    for _ in range(repeats):
        out = run([sys.executable, SEQ])
        t = float(re.search(r"SEQ_TOTAL\s+([0-9.]+)", out).group(1))
        best = t if best is None else min(best, t)
    return best


def measure_mpi(procs, repeats):
    best = None
    for _ in range(repeats):
        out = run(["mpiexec", "-n", str(procs), sys.executable, MPI])
        t = float(re.search(r"MPI_TOTAL\s+\d+\s+([0-9.]+)", out).group(1))
        best = t if best is None else min(best, t)
    return best


def main():
    cpu = os.cpu_count() or 4
    default_procs = [p for p in [1, 2, 4, 8, 16] if p <= cpu]

    parser = argparse.ArgumentParser()
    parser.add_argument("--procs", type=int, nargs="+", default=default_procs)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    print(f"Ядер CPU: {cpu} | процессы: {args.procs} | повторов: {args.repeats}\n")

    print("Замер последовательной версии ...")
    t_seq = measure_seq(args.repeats)
    print(f"  T_seq = {t_seq:.3f} c\n")

    rows = []
    for p in args.procs:
        print(f"Замер MPI (-n {p}) ...")
        t_p = measure_mpi(p, args.repeats)
        speedup = t_seq / t_p
        rows.append((p, t_p, speedup, speedup / p))
        print(f"  T_mpi({p}) = {t_p:.3f} c | S = {speedup:.2f}x | E = {speedup / p:.2f}\n")

    print("=" * 56)
    print(f"{'Процессы':>9} | {'Время, c':>10} | {'Ускорение':>10} | {'Эфф-сть':>8}")
    print("-" * 56)
    print(f"{'seq':>9} | {t_seq:>10.3f} | {'1.00x':>10} | {'-':>8}")
    for p, t_p, s, e in rows:
        print(f"{p:>9} | {t_p:>10.3f} | {s:>9.2f}x | {e:>8.2f}")
    print("=" * 56)

    csv_path = os.path.join(ROOT, "benchmark_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("procs,time_sec,speedup,efficiency\n")
        f.write(f"seq,{t_seq:.6f},1.0,\n")
        for p, t_p, s, e in rows:
            f.write(f"{p},{t_p:.6f},{s:.4f},{e:.4f}\n")
    print(f"\nРезультаты: {csv_path}")

    _maybe_plot(rows, ROOT)


def _maybe_plot(rows, root):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    procs = [r[0] for r in rows]
    speedup = [r[2] for r in rows]

    plt.figure(figsize=(7, 5))
    plt.plot(procs, procs, "--", color="gray", label="Идеальное ускорение")
    plt.plot(procs, speedup, "o-", color="#2b6cb0", label="Измеренное ускорение")
    plt.xlabel("Число MPI-процессов")
    plt.ylabel("Ускорение S(p) = T_seq / T_mpi(p)")
    plt.title("Ускорение распределённой рекомендательной системы (MPI)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = os.path.join(root, "speedup.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"График: {out}")


if __name__ == "__main__":
    main()
