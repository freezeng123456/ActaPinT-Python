#!/bin/bash
#SBATCH --job-name=actapint_seq_20261111
#SBATCH --partition=xhhctdnormal
#SBATCH --nodes=1
#SBATCH --nodelist=b02r06n03
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000M
#SBATCH --time=00:20:00
#SBATCH --output=/work/home/zenghang/actapint_20261111/sequential.out
#SBATCH --error=/work/home/zenghang/actapint_20261111/sequential.err
source /etc/profile
set -euo pipefail
ACTA_ROOT=/work/home/zenghang/actapint_20261111
ACTA_PY="$ACTA_ROOT/venv/bin/python"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p "$ACTA_ROOT/sequential"
trap 'date -u > "$ACTA_ROOT/sequential/failed"' ERR
cd "$ACTA_ROOT/code-sequential"
{
  date -u
  scontrol show job "$SLURM_JOB_ID"
  sha256sum "$ACTA_ROOT/source-sequential.tar.gz" actapint/stmg_fem.py actapint/stmg_sequential.py scripts/benchmark_stmg_sequential.py
} > "$ACTA_ROOT/sequential/provenance.txt"
for n in 2 4 8 16 32 512; do
  printf '%s %s\n' "$n" "$(date -u +%FT%TZ)" > "$ACTA_ROOT/sequential/current.txt"
  "$ACTA_PY" -m scripts.benchmark_stmg_sequential --steps "$n" --output "$ACTA_ROOT/sequential/steps-$n.json" > "$ACTA_ROOT/sequential/steps-$n.log" 2>&1
  "$ACTA_PY" -c 'import json,sys; x=json.load(open(sys.argv[1])); print(x["steps"],x["median_seconds"],flush=True)' "$ACTA_ROOT/sequential/steps-$n.json"
done
date -u > "$ACTA_ROOT/sequential/done"
