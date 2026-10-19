#!/usr/bin/env bash
# E13: per-config baseline / bmma / routerD / routerD+engage, MD5 check
DS=$1; TH=0.015
RES="TransactionSets/${DS}-${TH}00000=Results.txt"
RES="TransactionSets/${DS}-0.015000=Results.txt"
LOGD="../router_logs_v8"
gt() { grep computation_time "$RES" | awk '{print $2}'; }
echo "== E13 $DS =="
rm -f "$RES"; ./run/igcs_baseline128.exe "TransactionSets/${DS}" $TH >/dev/null 2>&1; echo "baseline: $(gt)"
rm -f "$RES"; ./run/igcs_bmma128.exe "TransactionSets/${DS}" $TH >/dev/null 2>&1; cp "$RES" /tmp/e13_${DS}_bmma.txt; echo "bmma: $(gt)"
rm -f "$RES"; env ROUTE_MODE=1 ROUTE_LOG="$LOGD/e13_${DS}.csv" ./run/igcs_router.exe "TransactionSets/${DS}" $TH >/dev/null 2>&1; echo "routerD: $(gt)"
python compare_mfis.py /tmp/e13_${DS}_bmma.txt "$RES" | tail -1
rm -f "$RES"; env ROUTE_MODE=1 ROUTE_ENGAGE_MS=2 ROUTE_LOG="$LOGD/e13_${DS}_eng2.csv" ./run/igcs_router.exe "TransactionSets/${DS}" $TH >/dev/null 2>&1; echo "routerD_eng2: $(gt)"
python compare_mfis.py /tmp/e13_${DS}_bmma.txt "$RES" | tail -1
