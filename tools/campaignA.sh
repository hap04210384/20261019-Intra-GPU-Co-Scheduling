#!/usr/bin/env bash
# Campaign A: interleaved medians, 12.8-rebuilt engines, router Mode D (+E)
# usage: bash campaignA.sh <tag> <dataset> <thresh> <rounds> [modes: bbtR default, bbtRE with pipe]
TAG=$1; DS=$2; TH=$3; ROUNDS=$4; MODES=${5:-bbtR}
THF=$(printf "%f" "$TH")
RES="TransactionSets/${DS}-${THF}=Results.txt"
OUT="../results_campaignA_${TAG}_${DS}.txt"
LOGDIR="../router_logs"; mkdir -p "$LOGDIR"
gettime() { grep "computation_time" "$RES" | awk '{print $2}'; }
run() {
  local exe=$1; shift
  rm -f "$RES"
  env "$@" "./run/${exe}" "TransactionSets/${DS}" "${TH}" >/dev/null 2>&1
  gettime
}
echo "== campaignA ${TAG} ${DS}@${TH} rounds=${ROUNDS} modes=${MODES} ==" > "$OUT"
Tb=""; Tt=""; TR=""; TE=""
for r in $(seq 1 "$ROUNDS"); do
  for m in $(echo "$MODES" | grep -o .); do
    case $m in
      b) t=$(run igcs_baseline128.exe); Tb="$Tb $t" ;;
      t) t=$(run igcs_bmma128.exe);     Tt="$Tt $t" ;;
      R) t=$(run igcs_router.exe ROUTE_MODE=1 ROUTE_LOG="${LOGDIR}/A_${TAG}_${DS}_r${r}.csv"); TR="$TR $t" ;;
      E) t=$(run igcs_router.exe ROUTE_MODE=2 ROUTE_LOG="${LOGDIR}/A_${TAG}_${DS}_E_r${r}.csv"); TE="$TE $t" ;;
    esac
    echo "round $r $m: $t" >> "$OUT"
  done
  if [ "$r" = "1" ]; then cp "$RES" /tmp/campA_base_${TAG}_${DS}.txt; fi
done
echo "-- medians --" >> "$OUT"
med() { echo $1 | tr ' ' '\n' | grep -v '^$' | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
[ -n "$Tb" ] && echo "baseline median: $(med "$Tb")  (runs:$Tb)" >> "$OUT"
[ -n "$Tt" ] && echo "bmma     median: $(med "$Tt")  (runs:$Tt)" >> "$OUT"
[ -n "$TR" ] && echo "routerD  median: $(med "$TR")  (runs:$TR)" >> "$OUT"
[ -n "$TE" ] && echo "routerE  median: $(med "$TE")  (runs:$TE)" >> "$OUT"
python compare_mfis.py /tmp/campA_base_${TAG}_${DS}.txt "$RES" >> "$OUT" 2>&1
tail -12 "$OUT"
