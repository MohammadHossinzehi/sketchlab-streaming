"""Cuckoo filter (partial-key cuckoo hashing) supporting insert, lookup and delete."""
import random
from .hashing import hash64


class CuckooFilter:
    def __init__(self, capacity=1024, bucket_size=4, fp_bits=12, max_kicks=500, seed=0):
        n = 1
        while n * bucket_size < capacity / 0.95:
            n <<= 1
        self.n, self.b, self.fp_bits, self.max_kicks = n, bucket_size, fp_bits, max_kicks
        self.buckets = [[] for _ in range(n)]
        self.size = 0
        self.rng = random.Random(seed)
        self.seed = seed

    def _fp(self, item):
        f = hash64(item, self.seed + 1) & ((1 << self.fp_bits) - 1)
        return f or 1

    def _i1(self, item):
        return hash64(item, self.seed) % self.n

    def _alt(self, i, fp):
        # n is always a power of two, so XOR with a masked hash is an involution
        return i ^ (hash64(fp, self.seed + 2) & (self.n - 1))

    def insert(self, item):
        fp, i1 = self._fp(item), self._i1(item)
        i2 = self._alt(i1, fp)
        for i in (i1, i2):
            if len(self.buckets[i]) < self.b:
                self.buckets[i].append(fp)
                self.size += 1
                return True
        i = self.rng.choice((i1, i2))
        trail = []
        for _ in range(self.max_kicks):
            k = self.rng.randrange(self.b)
            fp, self.buckets[i][k] = self.buckets[i][k], fp
            trail.append((i, k))
            i = self._alt(i, fp)
            if len(self.buckets[i]) < self.b:
                self.buckets[i].append(fp)
                self.size += 1
                return True
        for i, k in reversed(trail):  # roll back so a failed insert loses nothing
            fp, self.buckets[i][k] = self.buckets[i][k], fp
        return False

    def contains(self, item):
        fp, i1 = self._fp(item), self._i1(item)
        return fp in self.buckets[i1] or fp in self.buckets[self._alt(i1, fp)]

    def delete(self, item):
        fp, i1 = self._fp(item), self._i1(item)
        for i in (i1, self._alt(i1, fp)):
            if fp in self.buckets[i]:
                self.buckets[i].remove(fp)
                self.size -= 1
                return True
        return False

    def load_factor(self):
        return self.size / (self.n * self.b)

    def fp_rate_bound(self):
        return 2 * self.b / (1 << self.fp_bits)
