#!/usr/bin/env bash
# Run the final pipeline on Slurm. Each step waits for the steps it needs (--dependency=afterok).
#
# From comps1, in any folder:
#   bash pipeline/run.sh                  # everything, steps 1 to 10
#   bash pipeline/run.sh --step 3         # one step (the previous ones are assumed done)
#   bash pipeline/run.sh --from 5         # steps 5 to 10
#   bash pipeline/run.sh --dry-run        # print the sbatch commands, submit nothing
#
# Steps:  1 bank of blocks (CPU)          2 site selection + frozen scene list (CPU, network)
#         3 labels (CPU array, network)   4 Sentinel-2 stacks (CPU array, network)   [3 and 4 run in parallel]
#         5 usable sites, test set, folds (CPU)
#         6 step-count candidates (GPU array)   7 choose the step count (CPU)
#         8 cross-validation 5 folds x 5 seeds + 5 final models (GPU arrays)
#         9 ensembles, calibration, evaluation (CPU)   10 calibrated maps (CPU)
# Every step resumes: finished work (files, DONE markers, logged candidates) is skipped on a rerun.
set -euo pipefail

REPO=/h/460/tsalin/projects/MLOps/korotu-forest-density
RUN=/h/460/tsalin/projects/MLOps/slurm/run.sbatch
LOGS=/w/20261/tsalin/MLOps/slurm_logs
P=pipeline
NET_TASKS=20        # array tasks for labels and images (each takes every 20th site)
NET_PARALLEL=8      # at most this many of them at once (downloads)
GPU_PARALLEL=6      # GPU trainings at once
N_STEPS=$(cd "$REPO" && .venv/bin/python -I -c "import tomllib;print(len(tomllib.load(open('$P/config.toml','rb'))['train']['steps_candidates']))")
N_SEEDS=$(cd "$REPO" && .venv/bin/python -I -c "import tomllib;print(len(tomllib.load(open('$P/config.toml','rb'))['train']['seeds']))")
N_FOLDS=$(cd "$REPO" && .venv/bin/python -I -c "import tomllib;print(tomllib.load(open('$P/config.toml','rb'))['split']['folds'])")

cd "$REPO"
FROM=1; TO=10; DRY=0
while [ $# -gt 0 ]; do
    case $1 in
        --step) FROM=$2; TO=$2; shift 2 ;;
        --from) FROM=$2; shift 2 ;;
        --to) TO=$2; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done
want() { [ "$1" -ge "$FROM" ] && [ "$1" -le "$TO" ]; }
# output folders created once here, not by many array tasks at the same time (NFS race)
DATA=$(.venv/bin/python -I -c "import sys; sys.path.insert(0, '$P'); import common; print(common.DATA)")
[ "$DRY" = 1 ] || mkdir -p "$DATA"/{labels,stacks,chm,status/labels,status/images,predictions,models/logs,maps}

submit() {  # submit <name> "<jobs to wait for>" <sbatch options...> -- <script and arguments...>
    local name=$1 deps
    deps=$(echo $2)
    shift 2
    local opts=(--parsable --job-name="f-$name")
    while [ "$1" != "--" ]; do opts+=("$1"); shift; done
    shift
    if [ -n "$deps" ]; then opts+=(--dependency="afterok:${deps// /:}" --kill-on-invalid-dep=yes); fi
    local cmd=(sbatch "${opts[@]}" "$RUN" "$@")
    if [ "$DRY" = 1 ]; then
        echo "  [dry-run] step $STEP: ${cmd[*]}" >&2
        echo "<$name>"
        return
    fi
    command -v sbatch > /dev/null || { echo "sbatch not found: run this from comps1." >&2; exit 1; }
    local id
    id=$("${cmd[@]}")
    id=${id%%;*}
    echo "  step $STEP: $name -> job $id${deps:+ (waits for $deps)}" >&2
    echo "$id"
}
CPU=(-p cpunodes --gres=none)
ARR=(--output="$LOGS/%x-%A_%a.out")

J1= J2= J3= J4= J5= J6= J7= J8a= J8b= J9=
echo "Submitting steps $FROM to $TO:" >&2
STEP=1;  want 1  && J1=$(submit bank "" "${CPU[@]}" --cpus-per-task=2 --mem=16G --time=4:00:00 -- $P/01_bank.py)
STEP=2;  want 2  && J2=$(submit select "$J1" "${CPU[@]}" --cpus-per-task=8 --mem=16G --time=24:00:00 -- $P/02_select_sites.py)
STEP=3;  want 3  && J3=$(submit labels "$J2" "${CPU[@]}" "${ARR[@]}" --array=0-$((NET_TASKS - 1))%$NET_PARALLEL \
                         --cpus-per-task=4 --mem=16G --time=12:00:00 -- $P/03_labels.py)
STEP=4;  want 4  && J4=$(submit images "$J2" "${CPU[@]}" "${ARR[@]}" --array=0-$((NET_TASKS - 1))%$NET_PARALLEL \
                         --cpus-per-task=4 --mem=8G --time=12:00:00 -- $P/04_images.py)
STEP=5;  want 5  && J5=$(submit splits "$J3 $J4" "${CPU[@]}" --cpus-per-task=2 --mem=8G --time=1:00:00 -- $P/05_splits.py)
STEP=6;  want 6  && J6=$(submit steps "$J5" "${ARR[@]}" --array=0-$((N_STEPS - 1)) --cpus-per-task=8 --mem=32G \
                         --time=24:00:00 -- $P/06_train.py --mode steps)
STEP=7;  want 7  && J7=$(submit choose "$J6" "${CPU[@]}" --cpus-per-task=2 --mem=16G --time=1:00:00 -- $P/07_choose_steps.py)
if want 8; then
    STEP=8
    J8a=$(submit cv "$J7" "${ARR[@]}" --array=0-$((N_FOLDS * N_SEEDS - 1))%$GPU_PARALLEL --cpus-per-task=8 --mem=32G \
          --time=24:00:00 -- $P/06_train.py --mode cv)
    J8b=$(submit final "$J7" "${ARR[@]}" --array=0-$((N_SEEDS - 1)) --cpus-per-task=8 --mem=32G \
          --time=24:00:00 -- $P/06_train.py --mode final)
fi
STEP=9;  want 9  && J9=$(submit evaluate "$J8a $J8b" "${CPU[@]}" --cpus-per-task=4 --mem=48G --time=4:00:00 -- $P/08_evaluate.py)
STEP=10; want 10 && submit maps "$J9" "${CPU[@]}" --cpus-per-task=2 --mem=16G --time=4:00:00 -- $P/09_maps.py > /dev/null

echo "Follow: squeue -u \$USER   ·   logs: $LOGS/f-<name>-<job>[_<task>].out" >&2
