#!/bin/bash
#SBATCH --job-name=actapint_scaling_20261111
#SBATCH --partition=xhhctdnormal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000M
#SBATCH --time=01:20:00
#SBATCH --output=/work/home/zenghang/actapint_20261111/scaling.out
#SBATCH --error=/work/home/zenghang/actapint_20261111/scaling.err
source /etc/profile
set -euo pipefail
ACTA_ROOT=/work/home/zenghang/actapint_20261111
ACTA_PY="$ACTA_ROOT/venv/bin/python"
ACTA_MPI=/opt/hpc/software/mpi/hpcx/v2.11.0/gcc-7.3.1/bin/mpirun
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p "$ACTA_ROOT/scaling"
trap 'date -u > "$ACTA_ROOT/scaling/failed"' ERR
cd "$ACTA_ROOT/code"
{
  date -u
  scontrol show job "$SLURM_JOB_ID"
  lscpu
  sha256sum "$ACTA_ROOT/source.tar.gz" actapint/stmg_fem.py actapint/stmg_mpi.py scripts/benchmark_stmg_mpi.py
} > "$ACTA_ROOT/scaling/provenance.txt"
for mode in weak strong; do
  for p in 1 2 4 8 16; do
    if [ "$p" -eq 1 ]; then mapping=ppr:1:node; else mapping="ppr:$((p / 2)):node"; fi
    set -- "$ACTA_MPI" -n "$p" --map-by "$mapping" --bind-to core "$ACTA_PY" -m scripts.benchmark_stmg_mpi
    if [ "$mode" = strong ]; then set -- "$@" --steps 512; fi
    printf '%s %s %s\n' "$mode" "$p" "$(date -u +%FT%TZ)" > "$ACTA_ROOT/scaling/current.txt"
    "$@" --output "$ACTA_ROOT/scaling/$mode-p$p.json" > "$ACTA_ROOT/scaling/$mode-p$p.log" 2>&1
    "$ACTA_PY" -c 'import json,sys; x=json.load(open(sys.argv[1])); print(x["scaling"],x["ranks"],x["unique_hosts"],x["median_seconds"],flush=True)' "$ACTA_ROOT/scaling/$mode-p$p.json"
  done
done
date -u > "$ACTA_ROOT/scaling/done"
