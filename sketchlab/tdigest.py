"""Merging t-digest (k1 scale function) for streaming quantile estimates."""
import math


class TDigest:
    def __init__(self, delta=100):
        self.delta = delta
        self.centroids = []  # sorted list of [mean, weight]
        self.buf = []
        self.n = 0

    def add(self, x, w=1):
        self.buf.append((x, w))
        self.n += w
        if len(self.buf) >= 10 * self.delta:
            self._compress()

    def _k(self, q):
        return self.delta / (2 * math.pi) * math.asin(2 * q - 1)

    def _compress(self):
        pts = sorted(self.centroids + [list(p) for p in self.buf], key=lambda c: c[0])
        self.buf = []
        if not pts:
            return
        total = sum(w for _, w in pts)
        out = [list(pts[0])]
        seen = 0.0
        k_lo = self._k(0.0)
        for m, w in pts[1:]:
            cur = out[-1]
            q_hi = (seen + cur[1] + w) / total
            if self._k(min(q_hi, 1.0)) - k_lo <= 1:
                tw = cur[1] + w
                cur[0] += (m - cur[0]) * w / tw
                cur[1] = tw
            else:
                seen += cur[1]
                k_lo = self._k(seen / total)
                out.append([m, w])
        self.centroids = out

    def quantile(self, q):
        self._compress()
        c = self.centroids
        if not c:
            raise ValueError("empty digest")
        if len(c) == 1:
            return c[0][0]
        total = sum(w for _, w in c)
        target = q * total
        cum = 0.0
        for i, (m, w) in enumerate(c):
            mid = cum + w / 2
            if target < mid:
                if i == 0:
                    return m
                pm, pw = c[i - 1]
                pmid = cum - pw / 2
                return pm + (m - pm) * (target - pmid) / (mid - pmid)
            cum += w
        return c[-1][0]

    def cdf(self, x):
        self._compress()
        c = self.centroids
        total = sum(w for _, w in c)
        cum = 0.0
        for i, (m, w) in enumerate(c):
            if x < m:
                if i == 0:
                    return 0.0
                pm, pw = c[i - 1]
                pmid, mid = cum - pw / 2, cum + w / 2
                return (pmid + (mid - pmid) * (x - pm) / (m - pm)) / total
            cum += w
        return 1.0

    def merge(self, other):
        out = TDigest(self.delta)
        out.buf = [tuple(p) for p in self.centroids + other.centroids] + self.buf + other.buf
        out.n = self.n + other.n
        out._compress()
        return out
