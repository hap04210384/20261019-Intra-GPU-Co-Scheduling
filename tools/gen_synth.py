# Synthetic level-wise benchmark generator (E13/E14).
# Plants P disjoint pairs and P2 disjoint triples (each included whole with prob p);
# remaining items are rare fillers (below threshold). Result: a 3-level tree whose
# wave structure is controlled by (P, P2) independent of the item-space dimension d
# (fillers inflate d only). Item ids are dense (1..d), 0 avoided.
# Fast path: per-transaction included-pattern count ~ Poisson((P+P2)*p).
import random, sys, math

def poisson(lam, rng):
    L = math.exp(-lam); k = 0; prod = 1.0
    while True:
        prod *= rng.random()
        if prod <= L: return k
        k += 1

def gen(N, d, P, P2, p, seed, out):
    rng = random.Random(seed)
    npat = P + P2
    assert 2 * P + 3 * P2 <= d // 2, "patterns exceed half the item space"
    patterns = []
    nxt = 1
    for _ in range(P):
        patterns.append((nxt, nxt + 1)); nxt += 2
    for _ in range(P2):
        patterns.append((nxt, nxt + 1, nxt + 2)); nxt += 3
    pat_items = set(x for pat in patterns for x in pat)
    fillers = [i for i in range(1, d + 1) if i not in pat_items]
    noise_len = max(0, d // 128)          # per-item noise freq ~ noise_len/d < threshold
    lam = npat * p
    w1 = 2 * P + 3 * P2
    with open(out, "w") as f:
        for _ in range(N):
            row = []
            k = poisson(lam, rng)
            if k:
                for pi in rng.sample(range(npat), min(k, npat)):
                    row.extend(patterns[pi])
            for _ in range(noise_len):
                row.append(rng.choice(fillers))
            if not row:
                row = [rng.choice(fillers)]
            row.sort()
            f.write(" ".join(map(str, row)) + "\n")
    print(f"N={N} d={d} P={P} P2={P2} p={p} -> wave1={w1} cand_wave2=C({w1},2)={w1*(w1-1)//2} freq_pairs={P+3*P2} wave3={P2}")

if __name__ == "__main__":
    # args: N d P P2 p seed out
    gen(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
        float(sys.argv[5]), int(sys.argv[6]), sys.argv[7])
