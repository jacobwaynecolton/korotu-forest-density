#!/usr/bin/env bash
# Lance les 8 étapes de la vérification des modèles sur les sites NEON (États-Unis), sur Slurm.
# Chaque job attend que le précédent ait réussi (--dependency=afterok) : si la calibration (étape 1)
# échoue, rien d'autre ne tourne.
#
# À lancer sur comps1, depuis n'importe quel dossier :
#   bash independant-verification/model-verification-USA/lancer_usa.sh              # étapes 0 à 7
#   bash independant-verification/model-verification-USA/lancer_usa.sh --etape 3    # une seule étape (les précédentes sont supposées faites)
#   bash independant-verification/model-verification-USA/lancer_usa.sh --depuis 4   # de l'étape 4 à la 7
#   bash independant-verification/model-verification-USA/lancer_usa.sh --dry-run    # affiche les commandes sans rien soumettre
#
# Étapes : 0 effet de la résolution · 1 calibration sur l'Ontario · 2 blocs NEON ·
#          3 nuages de points -> CHM à 0,5 m · 4 labels · 5 images S2 · 6 prédire (GPU) · 7 évaluer
# Les étapes 4 à 6 exécutent les scripts de l'étude ontarienne (../model-verification), dirigés vers nos dossiers.
set -euo pipefail

PROJ=/h/460/tsalin/projects/MLOps/korotu-forest-density
RUN=/h/460/tsalin/projects/MLOps/slurm/run.sbatch
LOGS=/w/20261/tsalin/MLOps/slurm_logs
U=independant-verification/model-verification-USA

cd "$PROJ"

DEPUIS=0; JUSQUA=7; DRY=0
while [ $# -gt 0 ]; do
    case $1 in
        --etape) DEPUIS=$2; JUSQUA=$2; shift 2 ;;
        --depuis) DEPUIS=$2; shift 2 ;;
        --jusqua) JUSQUA=$2; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "Option inconnue : $1 (voir --help)" >&2; exit 2 ;;
    esac
done
faire() { [ "$1" -ge "$DEPUIS" ] && [ "$1" -le "$JUSQUA" ]; }

lancer() {  # lancer <nom> <cpu|gpu> "<jobs attendus>" <processeurs> <mémoire> <script et arguments...>
    local nom=$1 res=$2 deps cpus=$4 mem=$5
    deps=$(echo $3)
    shift 5
    # Pas d'option --export : avec --export=ALL,…, Slurm voulait recharger l'environnement de connexion et échouait.
    local opts=(--parsable --job-name="u-$nom" --cpus-per-task="$cpus" --mem="$mem")
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

if [ "$DRY" = 0 ] && { faire 2 || faire 3; } && [ ! -s "$HOME/.neon_token" ]; then
    echo "~/.neon_token absent ou vide : il faut ta clé d'API NEON (voir le README)." >&2
    exit 1
fi

J0= J1= J2= J3= J4= J5= J6= J7=
echo "Soumission des étapes $DEPUIS à $JUSQUA :" >&2
ETAPE=0; faire 0 && J0=$(lancer resolution cpu "" 2 16G $U/00_effet_resolution.py)
ETAPE=1; faire 1 && J1=$(lancer calibration cpu "$J0" 4 32G $U/01_calibrer_ontario.py)
ETAPE=2; faire 2 && J2=$(lancer sites cpu "$J1" 2 4G $U/02_choisir_sites.py)
ETAPE=3; faire 3 && J3=$(lancer nuages_chm cpu "$J2" 8 48G $U/03_nuages_chm.py 2)
ETAPE=4; faire 4 && J4=$(lancer labels cpu "$J3" 4 16G $U/04_labels.py)
ETAPE=5; faire 5 && J5=$(lancer images_s2 cpu "$J4" 8 16G $U/05_images_s2.py)
ETAPE=6; faire 6 && J6=$(lancer predire gpu "$J5" 8 32G $U/06_predire.py)
ETAPE=7; faire 7 && J7=$(lancer evaluer cpu "$J6" 2 16G $U/07_evaluer.py)

echo "Suivi : squeue -u \$USER   ·   logs : $LOGS/u-<nom>-<job>.out" >&2
