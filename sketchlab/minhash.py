"""MinHash signatures and an LSH banding index for Jaccard similarity search."""
from .hashing import hash64


class MinHash:
    def __init__(self, num_perm=128, seed=0):
        self.k, self.seed = num_perm, seed
        self.sig = [(1 << 64) - 1] * num_perm

    def add(self, item):
        for i in range(self.k):
            h = hash64(item, self.seed * 100003 + i)
            if h < self.sig[i]:
                self.sig[i] = h

    def update(self, items):
        for x in items:
            self.add(x)

    def jaccard(self, other):
        return sum(a == b for a, b in zip(self.sig, other.sig)) / self.k

    def merge(self, other):
        out = MinHash(self.k, self.seed)
        out.sig = [min(a, b) for a, b in zip(self.sig, other.sig)]
        return out


class LSHIndex:
    def __init__(self, bands=32, rows=4):
        self.bands, self.rows = bands, rows
        self.tables = [{} for _ in range(bands)]
        self.sigs = {}

    def insert(self, key, mh):
        assert mh.k == self.bands * self.rows
        self.sigs[key] = mh
        for b in range(self.bands):
            band = tuple(mh.sig[b * self.rows:(b + 1) * self.rows])
            self.tables[b].setdefault(band, set()).add(key)

    def query(self, mh, threshold=0.0):
        cands = set()
        for b in range(self.bands):
            band = tuple(mh.sig[b * self.rows:(b + 1) * self.rows])
            cands |= self.tables[b].get(band, set())
        res = [(k, mh.jaccard(self.sigs[k])) for k in cands]
        return sorted([r for r in res if r[1] >= threshold], key=lambda x: -x[1])

    def collision_probability(self, s):
        return 1 - (1 - s ** self.rows) ** self.bands
