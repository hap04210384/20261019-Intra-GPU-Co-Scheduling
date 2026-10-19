#!/usr/bin/env bash
DS=connect; TH=0.90
THF=$(printf "%f" "$TH")
RES="TransactionSets/${DS}.txt-${THF}=Results.txt"
med() { echo $1 | tr ' ' '\n' | grep -v '^$' | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
echo "== E11 KSCALE connect ==" >> ../results_E11.txt
for ks in 0.5 0.75 1.0 1.5 2.0 4.0; do
  T=""
  for r in 1 2 3; do
    rm -f "$RES"
    env ROUTE_MODE=1 ROUTE_KSCALE=$ks ./run/igcs_router.exe "TransactionSets/${DS}.txt" "${TH}" >/dev/null 2>&1
    t=$(grep "computation_time" "$RES" | awk '{print $2}')
    T="$T $t"
  done
  echo "KSCALE=$ks median=$(med "$T") runs:$T" >> ../results_E11.txt
  echo "done ks=$ks"
done
tail -8 ../results_E11.txt
