# sketchlab

Streaming probabilistic data structures from scratch in pure Python, with zero dependencies: **HyperLogLog**, **Count-Min sketch** (plain and conservative update, plus a heavy-hitter tracker), **Cuckoo filter**, **MinHash with an LSH index**, and a merging **t-digest**.

## Why

When a stream is too big to store, you trade exactness for a tiny, fixed memory footprint with provable error bounds. Each structure here answers a different question about a stream:

| Structure | Question | Memory | Error |
|---|---|---|---|
| `HyperLogLog` | How many distinct items? | 2^p registers | about 1.04 / sqrt(2^p) |
| `CountMin` | How often did X occur? | e/eps x ln(1/delta) counters | never under, over by at most eps x N with prob 1 - delta |
| `HeavyHitters` | Which items exceed a fraction phi of the stream? | one Count-Min plus candidates | same as Count-Min |
| `CuckooFilter` | Have I seen X (with deletes)? | fingerprints in buckets | false positive rate at most 2b / 2^f, no false negatives |
| `MinHash` / `LSHIndex` | How similar are two sets, and which stored sets are near this one? | k hashes per set | stderr about sqrt(J(1-J)/k) |
| `TDigest` | What is the p99 of this stream? | O(delta) centroids | finest accuracy in the tails |

Every structure that can be merged (HLL, Count-Min, MinHash, t-digest) supports `merge`, which makes them usable in map/reduce style pipelines.

## Run it

Requires Python 3.8+. No installs.

```
python -m unittest discover -s tests    # 18 tests
python bench.py                          # accuracy report
```

```python
from sketchlab import HyperLogLog, CountMin, CuckooFilter, MinHash, LSHIndex, TDigest

h = HyperLogLog(p=12)
for user in stream: h.add(user)
print(h.count())            # about 1.6% standard error

cf = CuckooFilter(capacity=100_000)
cf.insert("alice"); cf.contains("alice"); cf.delete("alice")

td = TDigest(delta=100)
for latency in latencies: td.add(latency)
print(td.quantile(0.99))
```

## Design notes

- **Hashing** (`hashing.py`): one seeded 64-bit hash (FNV-1a followed by a splitmix64 finalizer). Seeds derive independent hash functions for Count-Min rows and MinHash permutations, so results are deterministic and reproducible across runs and machines (unlike Python's salted `hash`).
- **HyperLogLog**: the top p bits choose a register, the rank of the first 1 bit in the remainder is stored. Small cardinalities use linear counting. Merge is an element-wise max, so merging equals sketching the union exactly (tested register for register).
- **Count-Min conservative update** only raises counters that are below the new minimum, which keeps the never-underestimate guarantee while cutting overcounting. Tests assert it is never worse than plain updates. The sketch built by `merge` is a plain sum, which stays valid.
- **Cuckoo filter** uses partial-key cuckoo hashing: the alternate bucket is `i XOR hash(fingerprint)`, so an item can be relocated knowing only its fingerprint. A failed insert after `max_kicks` rolls back its whole eviction chain, so a full filter never silently forgets a stored key (there is a test for this).
- **MinHash + LSH banding**: signatures split into b bands of r rows. Two sets with Jaccard similarity s collide in some band with probability 1 - (1 - s^r)^b, exposed as `collision_probability` so you can tune the threshold curve.
- **t-digest** uses the k1 scale function, which allows tiny clusters near q=0 and q=1 and large ones near the median, giving high relative accuracy at the tails. Quantiles interpolate between centroid midpoints.

## Testing

Tests check the properties each structure promises, not just example outputs: HLL error within 4 standard errors and exact merge equality, Count-Min never underestimates and stays within the eps bound for at least 98% of keys, Cuckoo has no false negatives through inserts and deletes and a measured false positive rate under the bound, MinHash estimates Jaccard within tolerance and LSH separates near from far sets, and t-digest quantiles on 50k Gaussian samples land within 0.06 of exact while using under 300 centroids.
