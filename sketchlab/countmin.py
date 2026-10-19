"""Count-Min sketch with optional conservative update, plus a heavy-hitter tracker."""
import math
from .hashing import hash64


class CountMin:
    def __init__(self, eps=0.001, delta=0.01, conservative=False, seed=0):
        self.w = math.ceil(math.e / eps)
        self.d = math.ceil(math.log(1 / delta))
        self.conservative, self.seed = conservative, seed
        self.t = [[0] * self.w for _ in range(self.d)]
        self.total = 0

    def _cells(self, item):
        return [(r, hash64(item, self.seed + r) % self.w) for r in range(self.d)]

    def add(self, item, c=1):
        cells = self._cells(item)
        self.total += c
        if self.conservative:
            target = min(self.t[r][j] for r, j in cells) + c
            for r, j in cells:
                if self.t[r][j] < target:
                    self.t[r][j] = target
        else:
            for r, j in cells:
                self.t[r][j] += c

    def estimate(self, item):
        return min(self.t[r][j] for r, j in self._cells(item))

    def merge(self, other):
        if (self.w, self.d, self.seed) != (other.w, other.d, other.seed):
            raise ValueError("incompatible sketches")
        out = CountMin(conservative=False, seed=self.seed)
        out.w, out.d = self.w, self.d
        out.t = [[a + b for a, b in zip(x, y)] for x, y in zip(self.t, other.t)]
        out.total = self.total + other.total
        return out


class HeavyHitters:
    """Tracks items whose estimated frequency exceeds phi * stream length."""

    def __init__(self, phi=0.01, eps=0.001, delta=0.01):
        self.phi = phi
        self.cm = CountMin(eps, delta, conservative=True)
        self.cands = {}

    def add(self, item):
        self.cm.add(item)
        est = self.cm.estimate(item)
        if est >= self.phi * self.cm.total:
            self.cands[item] = est
        n = self.cm.total
        for k in [k for k in self.cands if self.cm.estimate(k) < self.phi * n]:
            del self.cands[k]

    def top(self):
        return sorted(((k, self.cm.estimate(k)) for k in self.cands), key=lambda x: -x[1])
