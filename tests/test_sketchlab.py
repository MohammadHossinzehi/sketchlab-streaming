import math, random, unittest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sketchlab import *
from sketchlab.hashing import hash64


class TestHash(unittest.TestCase):
    def test_deterministic_and_seeded(self):
        self.assertEqual(hash64("a", 1), hash64("a", 1))
        self.assertNotEqual(hash64("a", 1), hash64("a", 2))
        self.assertNotEqual(hash64("a"), hash64("b"))


class TestHLL(unittest.TestCase):
    def test_accuracy(self):
        for n in (50, 5000, 100000):
            h = HyperLogLog(12)
            for i in range(n):
                h.add(i)
            self.assertLess(abs(h.count() - n) / n, 4 * h.std_error())

    def test_duplicates_ignored(self):
        h = HyperLogLog(10)
        for _ in range(5):
            for i in range(1000):
                h.add(i)
        self.assertLess(abs(h.count() - 1000) / 1000, 0.15)

    def test_merge_equals_union(self):
        a, b, u = HyperLogLog(10), HyperLogLog(10), HyperLogLog(10)
        for i in range(3000):
            a.add(i); u.add(i)
        for i in range(2000, 6000):
            b.add(i); u.add(i)
        self.assertEqual(a.merge(b).reg, u.reg)

    def test_incompatible(self):
        with self.assertRaises(ValueError):
            HyperLogLog(10).merge(HyperLogLog(11))


class TestCountMin(unittest.TestCase):
    def setUp(self):
        rng = random.Random(3)
        self.items = [int(rng.paretovariate(1.1)) for _ in range(20000)]
        self.truth = {}
        for x in self.items:
            self.truth[x] = self.truth.get(x, 0) + 1

    def test_never_underestimates_and_bounded(self):
        for cons in (False, True):
            cm = CountMin(0.01, 0.01, conservative=cons)
            for x in self.items:
                cm.add(x)
            bad = 0
            for k, v in self.truth.items():
                e = cm.estimate(k)
                self.assertGreaterEqual(e, v)
                bad += e - v > 0.01 * len(self.items)
            self.assertLessEqual(bad / len(self.truth), 0.02)

    def test_conservative_not_worse(self):
        a, b = CountMin(0.01, 0.01), CountMin(0.01, 0.01, conservative=True)
        for x in self.items:
            a.add(x); b.add(x)
        self.assertLessEqual(sum(b.estimate(k) - v for k, v in self.truth.items()),
                             sum(a.estimate(k) - v for k, v in self.truth.items()))

    def test_merge(self):
        a, b = CountMin(0.01, 0.1), CountMin(0.01, 0.1)
        a.add("x", 3); b.add("x", 4)
        self.assertEqual(a.merge(b).estimate("x"), 7)

    def test_heavy_hitters(self):
        hh = HeavyHitters(phi=0.05)
        for i in range(10000):
            hh.add("hot" if i % 5 == 0 else "cold%d" % i)
        self.assertEqual([k for k, _ in hh.top()], ["hot"])


class TestCuckoo(unittest.TestCase):
    def test_no_false_negatives_and_delete(self):
        cf = CuckooFilter(2000)
        keys = ["k%d" % i for i in range(1800)]
        for k in keys:
            self.assertTrue(cf.insert(k))
        self.assertTrue(all(cf.contains(k) for k in keys))
        for k in keys[:900]:
            self.assertTrue(cf.delete(k))
        self.assertEqual(cf.size, 900)
        self.assertTrue(all(cf.contains(k) for k in keys[900:]))

    def test_fp_rate(self):
        cf = CuckooFilter(5000)
        for i in range(4500):
            cf.insert("in%d" % i)
        fp = sum(cf.contains("out%d" % i) for i in range(20000)) / 20000
        self.assertLess(fp, cf.fp_rate_bound() * 2)

    def test_failed_insert_loses_nothing(self):
        cf = CuckooFilter(16, max_kicks=20)
        stored = []
        for i in range(200):
            if cf.insert("k%d" % i):
                stored.append("k%d" % i)
        self.assertLess(len(stored), 200)
        self.assertTrue(all(cf.contains(k) for k in stored))
        self.assertEqual(cf.size, len(stored))


class TestMinHash(unittest.TestCase):
    def test_jaccard_estimate(self):
        a, b = MinHash(256), MinHash(256)
        a.update(range(0, 1000)); b.update(range(500, 1500))
        self.assertAlmostEqual(a.jaccard(b), 1 / 3, delta=0.1)

    def test_merge_is_union(self):
        a, b, u = MinHash(64), MinHash(64), MinHash(64)
        a.update(range(100)); b.update(range(50, 200)); u.update(range(200))
        self.assertEqual(a.merge(b).sig, u.sig)

    def test_lsh(self):
        idx = LSHIndex(32, 4)
        base = set(range(300))
        for name, s in {"near": set(range(10, 310)), "far": set(range(1000, 1300))}.items():
            m = MinHash(128); m.update(s); idx.insert(name, m)
        q = MinHash(128); q.update(base)
        res = idx.query(q)
        self.assertEqual(res[0][0], "near")
        self.assertNotIn("far", [k for k, _ in res])
        self.assertGreater(idx.collision_probability(0.9), 0.99)
        self.assertLess(idx.collision_probability(0.1), 0.01)


class TestTDigest(unittest.TestCase):
    def test_quantiles(self):
        rng = random.Random(5)
        data = [rng.gauss(0, 1) for _ in range(50000)]
        td = TDigest(100)
        for x in data:
            td.add(x)
        s = sorted(data)
        for q in (0.01, 0.1, 0.5, 0.9, 0.99):
            self.assertLess(abs(td.quantile(q) - s[int(q * len(s))]), 0.06)
        self.assertLess(len(td.centroids), 300)

    def test_cdf_and_merge(self):
        a, b = TDigest(100), TDigest(100)
        for i in range(10000):
            a.add(i / 10000)
            b.add(1 + i / 10000)
        m = a.merge(b)
        self.assertAlmostEqual(m.quantile(0.5), 1.0, delta=0.03)
        self.assertAlmostEqual(m.cdf(1.5), 0.75, delta=0.03)
        self.assertEqual(m.n, 20000)

    def test_empty(self):
        with self.assertRaises(ValueError):
            TDigest().quantile(0.5)


if __name__ == "__main__":
    unittest.main()
