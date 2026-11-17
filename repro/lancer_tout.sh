#!/usr/bin/env bash
# Relance tout le pipeline sur Slurm pour reproduire les résultats du dépôt :
# CHM -> labels -> images S2 -> 3 U-Net -> ensemble -> Meta/SCANFI/NTEMS -> évaluation -> vérifications -> copie.
# Chaque job attend que ceux dont il dépend aient réussi (--dependency=afterok).
#
# À lancer sur comps1, sur une branche autre que main (les scripts réécrivent data/manifest/) :
#   bash repro/lancer_tout.sh                 # étapes 1 à 10
#   bash repro/lancer_tout.sh --etape 4       # une seule étape (les précédentes sont supposées faites)
#   bash repro/lancer_tout.sh --depuis 5      # de l'étape 5 à la 10
#   bash repro/lancer_tout.sh --dry-run       # affiche les commandes sbatch sans rien soumettre
#
# Étapes : 1 CHM · 2 labels · 3 contexte · 4 images S2 · 5 entraînement ×3 · 6 ensemble
#          7 Meta/SCANFI/NTEMS · 8 évaluation et analyses · 9 vérifications · 10 stockage et comparaison
set -euo pipefail

PROJ=/h/460/tsalin/projects/MLOps/korotu-forest-density
WORK=/w/20261/tsalin/MLOps
RUN=/h/460/tsalin/projects/MLOps/slurm/run.sbatch
LOGS=$WORK/slurm_logs
REF=${REF:-$WORK/reference_camarade}   # résultats du camarade, extraits de la branche main
OUT=${OUT:-$WORK/repro_resultats}      # copies horodatées de tes résultats
PY="$PROJ/.venv/bin/python -I"
META_INDEX=https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/dinov3_global_chm_v2_ml3/tiles.geojson
NTEMS_ZIP=https://opendata.nfis.org/downloads/forest_change/CA_canopy_cover_2022.zip
SOURCES="model:unet_r34_ens3 meta_chm_v2 meta_chm_v2_cal scanfi_v2 scanfi_v2_cal ntems_2022 ntems_2022_cal"
MODELES="unet_r34 unet_r34_s1 unet_r34_s2"

cd "$PROJ"

# ---------- modes internes, exécutés à l'intérieur d'un job ----------

telecharger() {  # telecharger <url> <fichier> : reprend un téléchargement interrompu, ne refait rien si le fichier existe
    [ -s "$2" ] && return 0
    curl -fL --retry 3 -C - -o "$2.part" "$1"
    mv "$2.part" "$2"
}

references() {  # étape 7a : index des tuiles Meta (absent du README) et carte NTEMS 2022
    mkdir -p data/raw/meta_chm data/raw/ntems
    telecharger "$META_INDEX" data/raw/meta_chm/tiles.geojson
    if [ -z "$(find data/raw/ntems -name '*.tif' -print -quit)" ]; then
        telecharger "$NTEMS_ZIP" data/raw/ntems/CA_canopy_cover_2022.zip
        $PY -m zipfile -e data/raw/ntems/CA_canopy_cover_2022.zip data/raw/ntems/
    fi
    echo "Rasters NTEMS (benchmarks.py utilise le premier de la liste) :"
    find data/raw/ntems -name '*.tif' | sort
}

stocker() {  # étape 10 : copie horodatée des résultats et comparaison avec le camarade
    local dest
    dest=$OUT/$(date +%Y%m%d-%H%M)
    mkdir -p "$dest/modeles"
    cp -r data/manifest verify/results "$dest/"
    cp -r report/figures "$dest/figures"
    for m in $MODELES; do
        if [ -d "models/$m" ]; then cp -r "models/$m" "$dest/modeles/"; fi
    done
    git diff --stat main -- data/manifest verify/results > "$dest/diff_vs_camarade.txt" || true
    $PY - "$REF" > "$dest/comparaison.txt" <<'EOF'
import csv, json, sys
from pathlib import Path

ref, root = Path(sys.argv[1]), Path.cwd()
lire = lambda p: json.loads(p.read_text()) if p.exists() else None
stats = lambda p: {r["site_id"]: r for r in csv.DictReader(open(p))} if p.exists() else {}
print(f"Comparaison avec les résultats du camarade (référence : {ref})\n")

a, b = stats(ref / "data/manifest/label_stats.csv"), stats(root / "data/manifest/label_stats.csv")
communs = sorted(set(a) & set(b))
if communs:
    ec = max(abs(float(a[s]["mean_cover_gt2m"]) - float(b[s]["mean_cover_gt2m"])) for s in communs)
    print(f"[2] labels : {len(b)} sites, écart max de densité moyenne par site = {ec:.2e} (0 = identique)")
    print(f"    sites seulement chez le camarade : {sorted(set(a) - set(b)) or 'aucun'}")
n = len(list((root / "data/pairs/stacks").glob("*_k*.tif")))
print(f"[4] images Sentinel-2 : {n} fichiers (attendu : 181)")
runs = {r["name"]: r for r in (lire(ref / "verify/results/training_runs.json") or [])}
for m in ("unet_r34", "unet_r34_s1", "unet_r34_s2"):
    c = lire(root / "models" / m / "config.json")
    if c and m in runs:
        print(f"[5] {m:12s} val {c['best_val_mae']:.4f} (camarade {runs[m]['val_mae']:.4f})   "
              f"test {c['test_mae']:.4f} (camarade {runs[m]['test_mae']:.4f})")
    else:
        print(f"[5] {m:12s} pas de config.json")
for sp in ("val", "test"):
    new, old = lire(root / f"data/manifest/eval_{sp}.json"), lire(ref / f"data/manifest/eval_{sp}.json") or []
    if not new:
        print(f"[8] eval_{sp}.json absent")
        continue
    o = {r["source"]: r for r in old if r["res"] == "10m"}
    for r in new:
        if r["res"] == "10m":
            c = o.get(r["source"])
            extra = f" (camarade {c['mae']:.4f}, écart {r['mae'] - c['mae']:+.4f})" if c else ""
            print(f"[8] {sp:4s} {r['source']:22s} MAE {r['mae']:.4f}{extra}")
s = lire(root / "verify/results/summary.json")
if s:
    print(f"[9] vérifications : {s['passed']}/{s['total']} réussies")
    for c in s["checks"]:
        if not c["ok"]:
            print("    ÉCHEC :", c["name"])
EOF
    cat "$dest/comparaison.txt"
    echo "Résultats copiés dans $dest"
}

if [ "${1:-}" = --interne ]; then
    case ${2:-} in
        references) references ;;
        stocker) stocker ;;
        *) echo "Mode interne inconnu : ${2:-}" >&2; exit 2 ;;
    esac
    exit 0
fi

# ---------- options ----------

DEPUIS=1; JUSQUA=10; DRY=0
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

lancer() {  # lancer <nom> <cpu|gpu> <afterok|afterany> "<jobs attendus>" <py|sh> <arguments...>
    local nom=$1 res=$2 type_dep=$3 deps type=$5
    deps=$(echo $4)
    shift 5
    local opts=(--parsable --job-name="k-$nom")
    if [ "$res" = cpu ]; then opts+=(-p cpunodes --gres=none); fi
    if [ -n "$deps" ]; then opts+=(--dependency="$type_dep:${deps// /:}" --kill-on-invalid-dep=yes); fi
    local cmd
    if [ "$type" = py ]; then
        cmd=(sbatch "${opts[@]}" "$RUN" "$@")
    else
        cmd=(sbatch "${opts[@]}" --output="$LOGS/%x-%j.out" --time=12:00:00 --mem=8G --wrap="$*")
    fi
    if [ "$DRY" = 1 ]; then
        echo "  [dry-run] étape $ETAPE · ${cmd[*]}" >&2
        echo "<$nom>"
        return
    fi
    local id
    id=$("${cmd[@]}")
    id=${id%%;*}
    echo "  étape $ETAPE · $nom -> job $id${deps:+ (attend $deps)}" >&2
    echo "$id"
}

# ---------- étape 0 : préparation (locale) ----------

if [ "$DRY" = 0 ]; then
    command -v sbatch > /dev/null || { echo "sbatch introuvable : lance ce script depuis comps1." >&2; exit 1; }
    if [ "$(git rev-parse --abbrev-ref HEAD)" = main ]; then
        echo "Tu es sur main. Crée d'abord une branche : git switch -c repro" >&2
        exit 1
    fi
    mkdir -p "$LOGS" "$OUT"
    if [ ! -d "$REF" ]; then
        mkdir -p "$REF"
        git archive main data/manifest verify/results | tar -x -C "$REF"
        echo "Référence du camarade extraite de main dans $REF" >&2
    fi
fi
n_sites=$(($(wc -l < data/manifest/sites.csv) - 1))
[ "$n_sites" = 49 ] || echo "Attention : sites.csv contient $n_sites sites (49 attendus)." >&2

J1= J2= J3= J4= J5a= J5b= J5c= J6= J7a= J7b= J7c= J8a= J8b= J8c= J8d= J9= J10=
echo "Soumission des étapes $DEPUIS à $JUSQUA :" >&2

ETAPE=1; faire 1 && J1=$(lancer chm cpu afterok "" py scripts/download_chm.py)
ETAPE=2; faire 2 && J2=$(lancer labels cpu afterok "$J1" py scripts/make_labels.py)
ETAPE=3; faire 3 && J3=$(lancer contexte cpu afterok "$J2" py scripts/context_layers.py)
ETAPE=4; faire 4 && J4=$(lancer images_s2 cpu afterok "$J2" py repro/stacks_from_list.py)

ETAPE=5
if faire 5; then
    J5a=$(lancer unet_s0 gpu afterok "$J4" py scripts/train_unet.py --name unet_r34 --seed 0 --steps 8000 --eval_every 500)
    J5b=$(lancer unet_s1 gpu afterok "$J4" py scripts/train_unet.py --name unet_r34_s1 --seed 1 --steps 8000 --eval_every 1000)
    J5c=$(lancer unet_s2 gpu afterok "$J4" py scripts/train_unet.py --name unet_r34_s2 --seed 2 --steps 8000 --eval_every 1000)
fi

ETAPE=6; faire 6 && J6=$(lancer ensemble cpu afterok "$J5a $J5b $J5c" py scripts/ensemble.py unet_r34_ens3 $MODELES)

ETAPE=7
if faire 7; then
    J7a=$(lancer references cpu afterok "" sh bash "$PROJ/repro/lancer_tout.sh" --interne references)
    J7b=$(lancer benchmarks cpu afterok "$J7a $J2" py scripts/benchmarks.py all meta_chm_v2 scanfi_v2 ntems_2022)
    J7c=$(lancer calibration cpu afterok "$J7b" py scripts/calibrate_benchmarks.py meta_chm_v2 scanfi_v2 ntems_2022)
fi

ETAPE=8
if faire 8; then
    # evaluate.py garde les anciennes lignes des sources qu'on ne lui redonne pas : on repart de fichiers vides
    for sp in val test; do
        f=data/manifest/eval_$sp.json
        if [ -f "$f" ] && git diff --quiet main -- "$f"; then
            if [ "$DRY" = 1 ]; then echo "  [dry-run] rm $f (version du camarade, gardée dans $REF)" >&2
            else rm "$f"; echo "  $f supprimé : c'était la version du camarade (copie dans $REF)" >&2; fi
        fi
    done
    J8a=$(lancer eval_val cpu afterok "$J6 $J7c" py scripts/evaluate.py val $SOURCES)
    J8b=$(lancer eval_test cpu afterok "$J6 $J7c" py scripts/evaluate.py test $SOURCES)
    J8c=$(lancer analyse_finale cpu afterok "$J6 $J7c $J3" py scripts/final_analysis.py unet_r34_ens3 ntems_2022)
    J8d=$(lancer incertitude cpu afterok "$J6" py scripts/uncertainty_preview.py unet_r34_ens3 $MODELES --site Z15_E0415_N5390_2020)
fi

ETAPE=9; faire 9 && J9=$(lancer verifications cpu afterok "$J8a $J8b $J6 $J7c $J4" py verify/run_all.py)
ETAPE=10; faire 10 && J10=$(lancer stockage cpu afterany "$J9 $J8a $J8b $J8c $J8d" sh bash "$PROJ/repro/lancer_tout.sh" --interne stocker)

echo "Suivi : squeue -u \$USER   ·   logs : $LOGS/k-<nom>-<job>.out" >&2
