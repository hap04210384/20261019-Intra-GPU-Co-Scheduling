#!/usr/bin/env bash
# 用法: run_pumsb_v9_round.sh <round>   —— 每轮跑一次（4模式交错），追加到 OUT
set -u
R="$1"
DS=pumsb_x256.txt; DSNAME=pumsb_x256; TH=0.80
THF=$(printf "%f" "$TH")
RES="TransactionSets/${DS}-${THF}=Results.txt"
OUT="E:/20261011-paper-Super processor/kimi_work/experiments/results/results_campaignA_v9_pumsb_x256.txt"
LOGDIR="E:/20261011-paper-Super processor/kimi_work/experiments/results/router_logs_v9"; mkdir -p "$LOGDIR"
if [ "$R" = "1" ]; then
  echo "== campaignA v9 ${DSNAME}@${TH} FULL per-round (baseline+bmma+routerD+routerE, $(date) ==" > "$OUT"
fi
rm -f "$RES"; ./run/igcs_baseline.exe "TransactionSets/${DS}" "${TH}" >/dev/null 2>&1; t=$(grep computation_time "$RES"|awk '{print $2}'); echo "round $R baseline: $t" >> "$OUT"
rm -f "$RES"; ./run/igcs_bmma.exe "TransactionSets/${DS}" "${TH}" >/dev/null 2>&1; t=$(grep computation_time "$RES"|awk '{print $2}'); echo "round $R bmma: $t" >> "$OUT"
rm -f "$RES"; env ROUTE_MODE=1 ROUTE_LOG="${LOGDIR}/A_v9_${DSNAME}_r${R}.csv" ./run/igcs_router.exe "TransactionSets/${DS}" "${TH}" >/dev/null 2>&1; t=$(grep computation_time "$RES"|awk '{print $2}'); echo "round $R D: $t" >> "$OUT"
rm -f "$RES"; env ROUTE_MODE=2 ROUTE_LOG="${LOGDIR}/A_v9_${DSNAME}_E_r${R}.csv" ./run/igcs_router.exe "TransactionSets/${DS}" "${TH}" >/dev/null 2>&1; t=$(grep computation_time "$RES"|awk '{print $2}'); echo "round $R E: $t" >> "$OUT"
echo "round $R done: $(grep "round $R " "$OUT" | tr '\n' ' ')"
