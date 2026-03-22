"""
Exact arithmetic in Z[phi] = {a + b*phi : a, b in Z}.

phi = (1 + sqrt(5)) / 2, the golden ratio.
phi^2 = phi + 1, so multiplication stays in the ring.
"""

PHI = (1 + 5**0.5) / 2


class ZPhi:
    """Exact element of Z[phi] stored as integer pair (a, b) representing a + b*phi."""

    __slots__ = ['a', 'b']

    def __init__(self, a=0, b=0):
        self.a = int(a)
        self.b = int(b)

    def __repr__(self):
        if self.b == 0:
            return f"{self.a}"
        elif self.a == 0:
            if self.b == 1:
                return "phi"
            elif self.b == -1:
                return "-phi"
            return f"{self.b}*phi"
        else:
            sign = "+" if self.b > 0 else "-"
            babs = abs(self.b)
            bstr = "" if babs == 1 else str(babs) + "*"
            return f"({self.a} {sign} {bstr}phi)"

    def to_float(self):
        return self.a + self.b * PHI

    def __add__(self, other):
        if isinstance(other, int):
            other = ZPhi(other, 0)
        return ZPhi(self.a + other.a, self.b + other.b)

    def __radd__(self, other):
        if isinstance(other, int):
            return ZPhi(self.a + other, self.b)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, int):
            other = ZPhi(other, 0)
        return ZPhi(self.a - other.a, self.b - other.b)

    def __neg__(self):
        return ZPhi(-self.a, -self.b)

    def __mul__(self, other):
        if isinstance(other, int):
            return ZPhi(self.a * other, self.b * other)
        if isinstance(other, ZPhi):
            return ZPhi(
                self.a * other.a + self.b * other.b,
                self.a * other.b + self.b * other.a + self.b * other.b,
            )
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, int):
            return ZPhi(self.a * other, self.b * other)
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, int):
            return self.a == other and self.b == 0
        if isinstance(other, ZPhi):
            return self.a == other.a and self.b == other.b
        return NotImplemented

    def __hash__(self):
        return hash((self.a, self.b))

    def galois_conjugate(self):
        """Galois automorphism: phi -> 1 - phi = -1/phi.
        a + b*phi -> (a + b) - b*phi
        """
        return ZPhi(self.a + self.b, -self.b)

    def norm(self):
        """Galois norm: z * bar(z). Always an integer for Z[phi]."""
        conj = self.galois_conjugate()
        prod = self * conj
        assert prod.b == 0, f"Norm not integer: {prod}"
        return prod.a


def quantize(x):
    """Find nearest Z[phi] point to a float x. Returns (ZPhi, error)."""
    best_a, best_b, best_err = 0, 0, abs(x)
    b_range = max(5, int(abs(x) / PHI) + 3)
    for b in range(-b_range, b_range + 1):
        a = round(x - b * PHI)
        err = abs(x - a - b * PHI)
        if err < best_err:
            best_a, best_b, best_err = a, b, err
    return ZPhi(best_a, best_b), best_err


def spectral_decompose(vec_zphi):
    """Decompose Z[phi]-valued vector into Galois eigenspaces.

    Returns (v_plus, v_minus) where v_minus is the Galois conjugate.
    """
    v_plus = vec_zphi
    v_minus = [z.galois_conjugate() for z in vec_zphi]
    return v_plus, v_minus
