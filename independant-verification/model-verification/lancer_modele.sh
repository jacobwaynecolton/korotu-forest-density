#!/usr/bin/env bash
# Lance les 6 étapes de la vérification des modèles sur de nouveaux sites, sur Slurm.
# Chaque job attend que le précédent ait réussi (--dependency=afterok).
#
# À lancer sur comps1, depuis n'importe quel dossier :
#   bash independant-verification/model-verification/lancer_modele.sh              # étapes 1 à 6
#   bash independant-verification/model-verification/lancer_modele.sh --etape 3    # une seule étape (les précédentes sont supposées faites)
#   bash independant-verification/model-verification/lancer_modele.sh --depuis 4   # de l'étape 4 à la 6
#   bash independant-verification/model-verification/lancer_modele.sh --dry-run    # affiche les commandes sans rien soumettre
#
# Étapes : 1 choisir les sites · 2 télécharger les CHM · 3 labels · 4 images S2 · 5 prédire (GPU) · 6 évaluer
set -euo pipefail

PROJ=/h/460/tsalin/projects/MLOps/korotu-forest-density
RUN=/h/460/tsalin/projects/MLOps/slurm/run.sbatch
LOGS=/w/20261/tsalin/MLOps/slurm_logs
M=independant-verification/model-verification

cd "$PROJ"

DEPUIS=1; JUSQUA=6; DRY=0
while [ $# -gt 0 ]; do
    case $1 in
        --etape) DEPUIS=$2; JUSQUA=$2; shift 2 ;;
        --depuis) DEPUIS=$2; shift 2 ;;
        --jusqua) JUSQUA=$2; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) echo "Option inconnue : $1 (voir --help)" >&2; exit 2 ;;
    esac
done
faire() { [ "$1" -ge "$DEPUIS" ] && [ "$1" -le "$JUSQUA" ]; }

lancer() {  # lancer <nom> <cpu|gpu> "<jobs attendus>" <processeurs> <mémoire> <script et arguments...>
    local nom=$1 res=$2 deps cpus=$4 mem=$5
    deps=$(echo $3)
    shift 5
    local opts=(--parsable --job-name="m-$nom" --cpus-per-task="$cpus" --mem="$mem")
    if [ "$res" = cpu ]; then opts+=(-p cpunodes --gres=none); fi   # sinon : partition GPU de run.sbatch
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

J1= J2= J3= J4= J5= J6=
echo "Soumission des étapes $DEPUIS à $JUSQUA :" >&2
ETAPE=1; faire 1 && J1=$(lancer sites cpu "" 2 8G $M/01_choisir_sites.py)
ETAPE=2; faire 2 && J2=$(lancer chm cpu "$J1" 8 8G $M/02_telecharger_chm.py 6)
ETAPE=3; faire 3 && J3=$(lancer labels cpu "$J2" 4 16G $M/03_labels.py)
ETAPE=4; faire 4 && J4=$(lancer images_s2 cpu "$J3" 8 16G $M/04_images_s2.py)
ETAPE=5; faire 5 && J5=$(lancer predire gpu "$J4" 8 32G $M/05_predire.py)
ETAPE=6; faire 6 && J6=$(lancer evaluer cpu "$J5" 2 16G $M/06_evaluer.py)

echo "Suivi : squeue -u \$USER   ·   logs : $LOGS/m-<nom>-<job>.out" >&2
