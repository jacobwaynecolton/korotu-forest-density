#!/usr/bin/env bash
# Lance les 4 étapes de la comparaison des labels avec GEDI sur Slurm.
# Chaque job attend que le précédent ait réussi (--dependency=afterok).
#
# À lancer sur comps1, depuis n'importe quel dossier :
#   bash independant-verification/gedi/lancer_gedi.sh                # étapes 1 à 4
#   bash independant-verification/gedi/lancer_gedi.sh --etape 2      # une seule étape (les précédentes sont supposées faites)
#   bash independant-verification/gedi/lancer_gedi.sh --depuis 3     # de l'étape 3 à la 4
#   bash independant-verification/gedi/lancer_gedi.sh --dry-run      # affiche les commandes sans rien soumettre
#
# Étapes : 1 lister les fichiers GEDI · 2 extraire les tirs · 3 références (labels, CHM, Hansen) · 4 comparer
set -euo pipefail

PROJ=/h/460/tsalin/projects/MLOps/korotu-forest-density
RUN=/h/460/tsalin/projects/MLOps/slurm/run.sbatch
LOGS=/w/20261/tsalin/MLOps/slurm_logs
G=independant-verification/gedi

cd "$PROJ"

DEPUIS=1; JUSQUA=4; DRY=0
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

lancer() {  # lancer <nom> "<jobs attendus>" <processeurs> <mémoire> <script et arguments...>
    local nom=$1 deps cpus=$3 mem=$4
    deps=$(echo $2)
    shift 4
    local opts=(--parsable --job-name="g-$nom" -p cpunodes --gres=none --cpus-per-task="$cpus" --mem="$mem")
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

if [ "$DRY" = 0 ] && [ ! -s "$HOME/.netrc" ]; then
    echo "~/.netrc absent ou vide : il faut tes identifiants NASA Earthdata (voir le README)." >&2
    exit 1
fi

J1= J2= J3= J4=
echo "Soumission des étapes $DEPUIS à $JUSQUA :" >&2
ETAPE=1; faire 1 && J1=$(lancer lister "" 1 4G $G/01_lister_fichiers.py)
ETAPE=2; faire 2 && J2=$(lancer extraire "$J1" 8 16G $G/02_extraire_tirs.py 6)
ETAPE=3; faire 3 && J3=$(lancer references "$J2" 4 32G $G/03_references.py)
ETAPE=4; faire 4 && J4=$(lancer comparer "$J3" 2 8G $G/04_comparer.py)

echo "Suivi : squeue -u \$USER   ·   logs : $LOGS/g-<nom>-<job>.out" >&2
