#!/usr/bin/env bash
DS=$1; RES="TransactionSets/${DS}-0.015000=Results.txt"
rm -f "$RES"; env ROUTE_MODE=1 ROUTE_LOG=../router_logs_v9/e13_${DS}.csv ./run/igcs_router.exe "TransactionSets/${DS}" 0.015 >/dev/null 2>&1
echo "routerD(final): $(grep computation_time "$RES"|awk '{print $2}')"
python compare_mfis.py /tmp/e13_${DS}_bmma.txt "$RES" 2>/dev/null | tail -1
