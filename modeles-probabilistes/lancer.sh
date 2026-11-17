#!/usr/bin/env bash
# Lance l'étude des modèles probabilistes sur Slurm : plis, poids SSL4EO, 85 entraînements (tableaux de tâches
# GPU), l'évaluation, les cartes finales et la comparaison aux cartes existantes. Chaque étape attend que les précédentes aient réussi (--dependency=afterok).
#
# À lancer sur comps1, depuis n'importe quel dossier :
#   bash modeles-probabilistes/lancer.sh              # tout
#   bash modeles-probabilistes/lancer.sh --etape 4    # seulement l'évaluation
#   bash modeles-probabilistes/lancer.sh --dry-run    # affiche les commandes sans rien soumettre
#
# Étapes : 1 plis · 2 poids SSL4EO · 3 entraînements (l1_r34_imagenet 25, q_r34_imagenet 30, q_r50_ssl4eo 30) · 4 évaluer
#          · 5 cartes finales · 6 comparaison à Meta, SCANFI et NTEMS
# Les entraînements déjà faits (fichier FAIT) sont sautés : relancer l'étape 3 ne refait que ce qui manque.
set -euo pipefail

PROJ=/h/460/tsalin/projects/MLOps/korotu-forest-density
RUN=/h/460/tsalin/projects/MLOps/slurm/run.sbatch
LOGS=/w/20261/tsalin/MLOps/slurm_logs
M=modeles-probabilistes
EN_PARALLELE=3   # tâches GPU simultanées par configuration

cd "$PROJ"

DEPUIS=1; JUSQUA=6; DRY=0
while [ $# -gt 0 ]; do
    case $1 in
        --etape) DEPUIS=$2; JUSQUA=$2; shift 2 ;;
        --depuis) DEPUIS=$2; shift 2 ;;
        --jusqua) JUSQUA=$2; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "Option inconnue : $1 (voir --help)" >&2; exit 2 ;;
    esac
done
faire() { [ "$1" -ge "$DEPUIS" ] && [ "$1" -le "$JUSQUA" ]; }

soumettre() {  # soumettre <nom> "<jobs attendus>" <options sbatch...> -- <script et arguments...>
    local nom=$1 deps
    deps=$(echo $2)
    shift 2
    local opts=(--parsable --job-name="p-$nom")
    while [ "$1" != "--" ]; do opts+=("$1"); shift; done
    shift
    if [ -n "$deps" ]; then opts+=(--dependency="afterok:${deps// /:}" --kill-on-invalid-dep=yes); fi
    local cmd=(sbatch "${opts[@]}" "$RUN" "$@")
    if [ "$DRY" = 1 ]; then
        echo "  [dry-run] étape $ETAPE · ${cmd[*]}" >&2
        echo "<$nom>"
        return
    fi
    command -v sbatch > /dev/null || { echo "sbatch introuvable : lance ce script depuis comps1." >&2; exit 1; }
    local id
    id=$("${cmd[@]}")
    id=${id%%;*}
    echo "  étape $ETAPE · $nom -> job $id${deps:+ (attend $deps)}" >&2
    echo "$id"
}
CPU=(-p cpunodes --gres=none)

J1= J2= J3a= J3b= J3c= J4= J5=
echo "Soumission des étapes $DEPUIS à $JUSQUA :" >&2
ETAPE=1; faire 1 && J1=$(soumettre plis "" "${CPU[@]}" --cpus-per-task=1 --mem=2G -- $M/01_plis.py)
ETAPE=2; faire 2 && J2=$(soumettre poids "" "${CPU[@]}" --cpus-per-task=1 --mem=4G -- $M/02_poids.py)
if faire 3; then
    ETAPE=3
    for spec in "l1_r34_imagenet 24" "q_r34_imagenet 29" "q_r50_ssl4eo 29"; do
        set -- $spec
        id=$(soumettre "$1" "$J1 $J2" --array="0-$2%$EN_PARALLELE" --cpus-per-task=4 --mem=24G \
             --output="$LOGS/%x-%A_%a.out" -- $M/03_entrainer.py --config "$1" --tache)
        case $1 in l1_*) J3a=$id ;; q_r34*) J3b=$id ;; q_r50*) J3c=$id ;; esac
    done
fi
ETAPE=4; faire 4 && J4=$(soumettre evaluer "$J3a $J3b $J3c" "${CPU[@]}" --cpus-per-task=4 --mem=32G -- $M/04_evaluer.py)
ETAPE=5; faire 5 && J5=$(soumettre cartes "$J4" "${CPU[@]}" --cpus-per-task=2 --mem=16G -- $M/05_cartes.py)
ETAPE=6; faire 6 && soumettre references "$J5" "${CPU[@]}" --cpus-per-task=2 --mem=16G -- $M/06_references.py > /dev/null

echo "Suivi : squeue -u \$USER   ·   journaux : $LOGS/p-<nom>-<job>[_<tâche>].out" >&2
