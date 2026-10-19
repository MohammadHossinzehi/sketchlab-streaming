"""Seeded 64-bit hashing (splitmix64 finalizer over FNV-1a) with no dependencies."""
MASK = (1 << 64) - 1


def _mix(z):
    z = (z + 0x9E3779B97F4A7C15) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def to_bytes(item):
    if isinstance(item, bytes):
        return item
    if isinstance(item, str):
        return item.encode("utf-8")
    return repr(item).encode("utf-8")


def hash64(item, seed=0):
    h = 0xCBF29CE484222325 ^ _mix(seed)
    for b in to_bytes(item):
        h = ((h ^ b) * 0x100000001B3) & MASK
    return _mix(h)
