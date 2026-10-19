"""HyperLogLog with small-range linear counting and a mergeable register array."""
import math
from .hashing import hash64


class HyperLogLog:
    def __init__(self, p=12, seed=0):
        if not 4 <= p <= 18:
            raise ValueError("p must be in [4, 18]")
        self.p, self.m, self.seed = p, 1 << p, seed
        self.reg = [0] * self.m

    def add(self, item):
        h = hash64(item, self.seed)
        idx = h >> (64 - self.p)
        rest = (h << self.p) & ((1 << 64) - 1)
        rank = (64 - self.p) + 1 if rest == 0 else 64 - rest.bit_length() + 1
        if rank > self.reg[idx]:
            self.reg[idx] = rank

    def _alpha(self):
        m = self.m
        if m == 16: return 0.673
        if m == 32: return 0.697
        if m == 64: return 0.709
        return 0.7213 / (1 + 1.079 / m)

    def count(self):
        m = self.m
        e = self._alpha() * m * m / sum(2.0 ** -r for r in self.reg)
        zeros = self.reg.count(0)
        if e <= 2.5 * m and zeros:
            return m * math.log(m / zeros)
        return e

    def merge(self, other):
        if (self.p, self.seed) != (other.p, other.seed):
            raise ValueError("incompatible sketches")
        out = HyperLogLog(self.p, self.seed)
        out.reg = [max(a, b) for a, b in zip(self.reg, other.reg)]
        return out

    def std_error(self):
        return 1.04 / math.sqrt(self.m)
