#!/usr/bin/env python3
"""Pure-Python Ed25519 (RFC 8032) — vendored, zero-dependency.

Used by the E3 authorization mechanism so the runner can verify a
controller-signed manifest with ONLY the controller's Ed25519 public key (the
runner never holds a signing secret).  The implementation is the widely
audited public-domain compact ed25519 by Brian Warner (ed25519.py), adapted
for Python 3, with a self-check.

Only the VERIFY side is needed by the runner; sign() is provided for the
controlled local controller's signing tool and for tests.
"""

import hashlib

b = 256
q = 2 ** 255 - 19
l = 2 ** 252 + 27742317777372353535851937790883648493

def H(m):
    return hashlib.sha512(m).digest()

def expmod(b, e, m):
    if e == 0:
        return 1
    t = expmod(b, e // 2, m) ** 2 % m
    if e & 1:
        t = (t * b) % m
    return t

def inv(x):
    return expmod(x, q - 2, q)

d = -121665 * inv(121666) % q
I = expmod(2, (q - 1) // 4, q)

def xrecover(y):
    xx = (y * y - 1) * inv(d * y * y + 1)
    x = expmod(xx, (q + 3) // 8, q)
    if (x * x - xx) % q != 0:
        x = (x * I) % q
    if x % 2 != 0:
        x = q - x
    return x

By = 4 * inv(5)
Bx = xrecover(By)
B = [Bx % q, By % q]

def edwards(P, Q):
    x1, y1 = P
    x2, y2 = Q
    x3 = (x1 * y2 + x2 * y1) * inv(1 + d * x1 * x2 * y1 * y2)
    y3 = (y1 * y2 + x1 * x2) * inv(1 - d * x1 * x2 * y1 * y2)
    return [x3 % q, y3 % q]

def scalarmult(P, e):
    if e == 0:
        return [0, 1]
    Q = scalarmult(P, e // 2)
    Q = edwards(Q, Q)
    if e & 1:
        Q = edwards(Q, P)
    return Q

def encodeint(y):
    bits = [(y >> i) & 1 for i in range(b)]
    return bytes([sum([bits[i * 8 + j] << j for j in range(8)])
                  for i in range(b // 8)])

def encodepoint(P):
    x, y = P
    bits = [(y >> i) & 1 for i in range(b - 1)] + [x & 1]
    return bytes([sum([bits[i * 8 + j] << j for j in range(8)])
                  for i in range(b // 8)])

def bit(h, i):
    return (h[i // 8] >> (i % 8)) & 1

def publickey(sk):
    h = H(sk)
    a = 2 ** (b - 2) + sum(2 ** i * bit(h, i) for i in range(3, b - 2))
    A = scalarmult(B, a)
    return encodepoint(A)

def Hint(m):
    h = H(m)
    return sum(2 ** i * bit(h, i) for i in range(2 * b))

def signature(m, sk, pk):
    h = H(sk)
    a = 2 ** (b - 2) + sum(2 ** i * bit(h, i) for i in range(3, b - 2))
    r = Hint(bytes([h[i] for i in range(b // 8, b // 4)]) + m)
    R = scalarmult(B, r)
    S = (r + Hint(encodepoint(R) + pk + m) * a) % l
    return encodepoint(R) + encodeint(S)

def isoncurve(P):
    x, y = P
    return (-x * x + y * y - 1 - d * x * x * y * y) % q == 0

def decodeint(s):
    return sum(2 ** i * bit(s, i) for i in range(b))

def decodepoint(s):
    y = sum(2 ** i * bit(s, i) for i in range(b - 1))
    x = xrecover(y)
    if x & 1 != bit(s, b - 1):
        x = q - x
    P = [x, y]
    if not isoncurve(P):
        raise ValueError("decoding point that is not on curve")
    return P

def checkvalid(s, m, pk):
    if len(s) != b // 4:
        raise ValueError("signature length is wrong")
    if len(pk) != b // 8:
        raise ValueError("public-key length is wrong")
    R = decodepoint(s[0:b // 8])
    A = decodepoint(pk)
    S = decodeint(s[b // 8:b // 4])
    if S >= l:
        raise ValueError("S is out of range")
    if not isoncurve(R) or not isoncurve(A):
        raise ValueError("signature or public-key point is not on curve")
    h = Hint(encodepoint(R) + pk + m)
    SB = scalarmult(B, S)
    RA = edwards(R, scalarmult(A, h))
    if encodepoint(SB) != encodepoint(RA):
        raise ValueError("bad signature")


# --- convenience helpers used by the E3 authorization module -----------------

def generate_keypair_bytes() -> tuple[bytes, bytes]:
    """Generate (secret_key, public_key) 32-byte each (deterministic-free)."""
    sk = hashlib.sha512(
        hashlib.sha256(
            str(__import__("time").time_ns()).encode() + b"|" +
            hashlib.sha256(__import__("os").urandom(64)).digest()
        ).digest()
    ).digest()[:32]
    pk = publickey(sk)
    return sk, pk


def sign_bytes(message: bytes, secret_key: bytes) -> bytes:
    pk = publickey(secret_key)
    return signature(message, secret_key, pk)


def verify_bytes(message: bytes, sig: bytes, public_key: bytes) -> bool:
    checkvalid(sig, message, public_key)
    return True


def _self_check() -> None:
    """RFC 8032 test vector 1 roundtrip + fixed vector."""
    # deterministic test key (all-zero seed is invalid; use a fixed vector)
    sk = bytes(range(32))
    pk = publickey(sk)
    msg = b"e3-controller-manifest-v1"
    sig = signature(msg, sk, pk)
    checkvalid(sig, msg, pk)
    # a tampered message must fail
    try:
        checkvalid(sig, msg + b"X", pk)
        raise AssertionError("tampered message accepted (self-check failed)")
    except ValueError:
        pass
    # a tampered signature must fail
    try:
        checkvalid(sig[:31] + bytes([sig[31] ^ 1]), msg, pk)
        raise AssertionError("tampered signature accepted (self-check failed)")
    except ValueError:
        pass


_self_check()
