from fractions import Fraction as Frac


ATOMIC_AM_SYMBOLS = 'spdfghiklmnoqrtuvwxyz'
ATOMIC_AM_SYMBOLS_UP = 'SPDFGHIKLMNOQRTUVWXYZ'
DIATOMIC_AM_SYMBOLS = 'σπδφγηικμνo'
DIATOMIC_AM_SYMBOLS_UP = 'ΣΠ∆ΦΓHIKMNOQRTUVWXYZ'


class Orbital:
    def __init__(self, n, l, ml, spin):
        """An orbital
        :param n: shell
        :param l: angular momentum
        :param ml: projected angular momentum (azimuthal, etc.)
        :param spin: electron spin
        """
        if not isinstance(n, int) or n < 1:
            raise SyntaxError("Shells (n) must be integers greater than 0, got: n = {n}")

        if not isinstance(l, int) or l < 0:
            raise SyntaxError(f"Angular momentum (l) must be an integer >= 0, got: l = {l}")

        if not isinstance(ml, int) or ml < -l or ml > l:
            raise SyntaxError("Projected angular momentum (ml) must be an integer such that -l <= ml <= l,"
                              f"got: ml = {ml}, l = {l}")

        if spin == 1 or spin == Frac(1, 2) or spin == 'alpha':
            self.spin = Frac(1, 2)
        elif spin == -1 or spin == -Frac(1, 2) or spin == 'beta':
            self.spin = -Frac(1, 2)
        else:
            raise SyntaxError("Invalid spin")

        self.n = n
        self.l = l
        self.ml = ml

    def __str__(self):
        spin = 'a' if self.spin > 0 else 'b'
        return self.spatial_str() + spin

    def __eq__(self, o):
        if type(self) == type(o) \
                and self.n == o.n and self.l == o.l \
                and self.ml == o.ml and self.spin == o.spin:
            return True
        return False

    def __repr__(self):
        return str(self)

    def spatial_str(self):
        base = f'{self.n}{self.orb_symbol}'
        if self.l == 0:
            return base
        return f'{base}_{{{self.ml}}}'


class AtomicSpinOrbital(Orbital):
    """An atomic spin orbital"""

    def __init__(self, n, l, ml, spin):
        if isinstance(l, str):
            if l in ATOMIC_AM_SYMBOLS:
                l = ATOMIC_AM_SYMBOLS.index(l)
            else:
                raise SyntaxError("Invalid orbital")

        # Atomic orbitals of higher angular momentum start at n = l + 1
        if n - l < 1:
            raise SyntaxError("Invalid subshell")

        super().__init__(n, l, ml, spin)
        self.orb_symbol = ATOMIC_AM_SYMBOLS[l]


class DiatomicSpinOrbital(Orbital):
    """Due to the limitations of typing Greek characters on a latin keyboard,
    The latin equivalents will be accepted and used for variable names.
    Furthermore, due to code conventions encouraging lowercase for variable
    names, the lowercase form will be used
    """

    def __init__(self, n, l, ml, spin):
        if isinstance(l, str):
            if l in DIATOMIC_AM_SYMBOLS:
                l = DIATOMIC_AM_SYMBOLS.index(l)
            elif l in ATOMIC_AM_SYMBOLS:
                # Allow the latin equivalent
                l = ATOMIC_AM_SYMBOLS.index(l)
            else:
                raise SyntaxError("Invalid orbital")

        super().__init__(n, l, ml, spin)
        if not abs(ml) == l:
            raise SyntaxError(f"Diatomic orbitals may only have ml = ±l, got: ml = {ml}, l = {l}")
        self.orb_symbol = DIATOMIC_AM_SYMBOLS[l]
