#!/usr/bin/env bash
# v9: router-only rerun (baseline/bmma unchanged from v8)
TAG=v9; DS=$1; TH=$2; ROUNDS=$3
THF=$(printf "%f" "$TH")
RES="TransactionSets/${DS}-${THF}=Results.txt"
OUT="../results_campaignA_${TAG}_${DS}.txt"
LOGDIR="../router_logs_${TAG}"; mkdir -p "$LOGDIR"
med() { echo $1 | tr ' ' '\n' | grep -v '^$' | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
TR=""; TE=""
echo "== campaignA v9 ${DS}@${TH} ==" > "$OUT"
for r in $(seq 1 "$ROUNDS"); do
  rm -f "$RES"; env ROUTE_MODE=1 ROUTE_LOG="${LOGDIR}/A_v9_${DS}_r${r}.csv" ./run/igcs_router.exe "TransactionSets/${DS}" "${TH}" >/dev/null 2>&1; t=$(grep computation_time "$RES"|awk '{print $2}'); TR="$TR $t"; echo "round $r D: $t" >> "$OUT"
  rm -f "$RES"; env ROUTE_MODE=2 ROUTE_LOG="${LOGDIR}/A_v9_${DS}_E_r${r}.csv" ./run/igcs_router.exe "TransactionSets/${DS}" "${TH}" >/dev/null 2>&1; t=$(grep computation_time "$RES"|awk '{print $2}'); TE="$TE $t"; echo "round $r E: $t" >> "$OUT"
done
echo "routerD median: $(med "$TR")  (runs:$TR)" >> "$OUT"
echo "routerE median: $(med "$TE")  (runs:$TE)" >> "$OUT"
tail -4 "$OUT"
