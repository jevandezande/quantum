from dataclasses import dataclass
from fractions import Fraction as Frac
from typing import TypeAlias, Literal, Optional


ATOMIC_AM_SYMBOLS = 'spdfghiklmnoqrtuvwxyz'
ATOMIC_AM_SYMBOLS_UP = ATOMIC_AM_SYMBOLS.upper()
DIATOMIC_AM_SYMBOLS = 'σπδφγηικμνoqrtuvwxyz'
DIATOMIC_AM_SYMBOLS_UP = DIATOMIC_AM_SYMBOLS.upper()


@dataclass
class Orbital:
    """
    An orbital
    :param n: shell
    :param l: angular momentum
    :param ml: projected angular momentum (azimuthal, etc.)
    :param spin: electron spin
    """
    n: int
    l: int
    ml: int
    spin: int | Frac
    orb_symbol: Optional[str] = None

    def __post_init__(self):
        if self.n < 1:
            raise ValueError("Shells (n) must be integers greater than 0, got: {self.n}")
        if self.l < 0:
            raise ValueError(f"Angular momentum (l) must be an integer >= 0, got: {self.l}")
        if self.ml < -self.l or self.ml > self.l:
            raise ValueError("Projected angular momentum (ml) must be an integer such that -l <= ml <= l, "
                             f"got: {self.l}, {self.ml=}")

        up = [1, Frac(1, 2), 'alpha', 'α']
        down = [-1, -Frac(1, 2), 'beta', 'β']
        if self.spin in up:
            self.spin = Frac(1, 2)
        elif self.spin in down:
            self.spin = -Frac(1, 2)
        else:
            raise ValueError(f"Spin must be in {up=} or {down=}, got: {self.spin}")

    def __str__(self):
        spin = 'a' if self.spin > 0 else 'b'
        return self.spatial_str() + spin

    def __repr__(self):
        return str(self)

    def spatial_str(self):
        base = f'{self.n}{self.orb_symbol}'
        if self.l == 0:
            return base
        return f'{base}_{{{self.ml}}}'


class AtomicSpinOrbital(Orbital):
    def __init__(self, n, l, ml, spin):
        """
        An atomic spin orbital
        """
        if isinstance(l, str):
            if l in ATOMIC_AM_SYMBOLS:
                l = ATOMIC_AM_SYMBOLS.index(l)
            else:
                raise ValueError(f"Invalid orbital angular momentum: {l}")

        # Atomic orbitals of higher angular momentum start at n = l + 1
        if n - l < 1:
            raise ValueError(f"Invalid orbital subshell: {l}")

        super().__init__(n, l, ml, spin)
        self.orb_symbol = ATOMIC_AM_SYMBOLS[l]


class DiatomicSpinOrbital(Orbital):
    def __init__(self, n, l, ml, spin):
        """
        A diatomic spinorbital

        Diatomic orbitals utilize Greek letters.
        Latin equivalents will be automatically converted upon initialization.
        """
        if isinstance(l, str):
            if l in DIATOMIC_AM_SYMBOLS:
                l = DIATOMIC_AM_SYMBOLS.index(l)
            elif l in ATOMIC_AM_SYMBOLS:
                # Allow the Latin equivalent
                l = ATOMIC_AM_SYMBOLS.index(l)
            else:
                raise ValueError(f"Invalid orbital angular momentum: {l}")

        super().__init__(n, l, ml, spin)
        if not abs(ml) == l:
            raise ValueError(f"Diatomic orbitals may only have ml = ±l, got: l = {l}, ml = {ml}")
        self.orb_symbol = DIATOMIC_AM_SYMBOLS[l]
