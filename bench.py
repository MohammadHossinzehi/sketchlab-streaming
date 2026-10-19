"""Accuracy report: python bench.py"""
import random
from sketchlab import *

def main():
    rng = random.Random(1)
    print("HyperLogLog (p=12, theoretical stderr %.2f%%)" % (100 * HyperLogLog(12).std_error()))
    for n in (100, 10_000, 200_000):
        h = HyperLogLog(12)
        for i in range(n):
            h.add("u%d" % i)
        c = h.count()
        print("  n=%-7d est=%-9.0f err=%.2f%%" % (n, c, 100 * abs(c - n) / n))
    print("Count-Min, zipf stream, plain vs conservative (mean overcount on 200 items)")
    items = [int(rng.paretovariate(1.2)) for _ in range(50_000)]
    truth = {}
    for x in items:
        truth[x] = truth.get(x, 0) + 1
    for cons in (False, True):
        cm = CountMin(0.05, 0.01, conservative=cons)
        for x in items:
            cm.add(x)
        keys = list(truth)[:200]
        print("  conservative=%-5s mean overcount=%.2f" % (cons, sum(cm.estimate(k) - truth[k] for k in keys) / len(keys)))
    cf = CuckooFilter(10_000)
    ins = sum(cf.insert("k%d" % i) for i in range(10_000))
    fp = sum(cf.contains("x%d" % i) for i in range(20_000)) / 20_000
    print("Cuckoo: inserted %d, load %.2f, measured FPR %.4f (bound %.4f)" % (ins, cf.load_factor(), fp, cf.fp_rate_bound()))
    td = TDigest(100)
    data = [rng.gauss(0, 1) for _ in range(100_000)]
    for x in data:
        td.add(x)
    s = sorted(data)
    for q in (0.01, 0.5, 0.99, 0.999):
        print("  t-digest q=%.3f est=%.4f exact=%.4f (%d centroids)" % (q, td.quantile(q), s[int(q * len(s))], len(td.centroids)))

if __name__ == "__main__":
    main()
