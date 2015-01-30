from itertools import combinations
from fractions import Fraction as Frac
import numpy as np


class SpinOrbital:
    def __init__(self, n, l, ml, spin):
        """A spin orbital"""
        self.orbs = 'spdfghklmn'
        if not isinstance(n, int) or n < 1:
            raise SyntaxError("Shells are integers starting at 1")

        if isinstance(l, str):
            try:
                l = self.orbs.index(l)
            except ValueError:
                raise SyntaxError("Invalid orbital")
        elif not isinstance(l, int) or l < 0:
            raise SyntaxError("Orbitals are ints >=0 or the appropriate string")
        if n - l < 1:
            raise SyntaxError("Invalid subshell")

        if ml > l or not isinstance(ml, int):
            raise SyntaxError("Azimuthal angular momentum (ml) is a positive " +
                              "integer such that -l <= ml <= l")

        if spin == 1 or spin == Frac(1/2) or spin == 'alpha':
            self.spin = Frac(1/2)
        elif spin == -1 or spin == Frac(-1/2) or spin == 'beta':
            self.spin = Frac(-1/2)
        else:
            raise SyntaxError("Invalid spin")

        self.n = n
        self.l = l
        self.ml = ml

    def __str__(self):
        spin = 'a' if self.spin > 0 else 'b'
        orb = self.orbs[self.l]
        return '{n}{l}_{{{ml}}}{spin}'.format(n=self.n, l=orb, ml=self.ml,
                                              spin=spin)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, o):
        if self.n == o.n and self.l == o.l\
                and self.ml == o.l and self.spin == o.spin:
            return True
        return False

def subshell_iterator(shell, l):
    """Iterate through all the orbitals in the specified subshell"""
    # iterate from highest to lowest m_l

def spin_iterator():
    """Iterate through alpha (1/2) and beta (-1/2) spins"""
    yield Frac(1, 2)
    yield -Frac(1, 2)


def spin_orbitals_iterator(shell, l):
    for ml in range(l, -l-1, -1):
        for spin in [Frac(1, 2), Frac(-1, 2)]:
            yield SpinOrbital(shell, l, ml, spin)


def occupy(shell, l, e_num):
    """Make an iterator of all possible combinations"""
    return combinations(spin_orbitals_iterator(shell, l), e_num)


def calc_vals(orbs):
    """Calculate the total angular momentum and spin
    WARNING: It does not check if the set of orbitals is valid
    """
    ang = 0
    spin = 0
    for orb in orbs:
        ang += orb.ml
        spin += orb.spin

    return ang, spin


class TermSymbol:
    def __init__(self, am, mult):
        self.am = am
        self.mult = mult

    def __str__(self):
        symbols = 'SPDFGHKLMN'
        return '{}{}'.format(self.mult, symbols[abs(self.am)])

    def __repr__(self):
        return str(self)



def term_symbol(ang, spin):
    """Write the associated term symbol for the given state"""
    mult = 2*abs(spin) + 1
    symbols = 'SPDFGHKLMN'

    return '{}{}'.format(mult, symbols[abs(ang)])


def find_term_symbol(orbs, latex=False):
    """Generate the term symbol for the given set of orbitals
    WARNING: It does not check if the set of orbitals is valid
    """
    ang, spin = calc_vals(orbs)
    return term_symbol(ang, spin)


class TermTable:
    def __init__(self, max_am, max_spin):
        self.max_am = max_am
        self.width = max_am + 1
        self.max_spin = max_spin
        self.min_spin = max_spin % 1
        self.height = int(max_spin) + 1
        self.table = np.zeros((self.height, self.width))
        # TODO: add combinations to each term
        self.combs = {}

    def __str__(self):
        """Flip the table for printing"""
        return str(self.table[::-1])

    def __eq__(self, o):
        if self.max_am != o.max_am or self.max_spin != o.max_spin:
            return False
        if not (self.table == o.table).all():
            return False
        return True

    def get(self, am, spin):
        return self.table[int(spin)][am]

    def set(self, am, spin, val):
        self.table[int(spin)][am] = val

    def increment(self, am, spin):
        self.table[int(spin)][am] += 1

    def clean_table(self):
        """Remove all the lower manifestations of terms
        i.e. subtract the value of (a,b) from other terms where x<=a, y<=b"""
        for i in reversed(range(len(self.table))):
            for j in reversed(range(len(self.table[0]))):
                count = self.table[i, j]
                self.table[:i+1, :j+1] -= count
                self.table[i, j] = count


def subshell_terms(shell, l, e_num):
    max_am = l*e_num
    if e_num <= 2*l + 1:
        max_spin = Frac(e_num, 2)
    else:
        max_spin = Frac(4*l+2 - e_num)
    t = TermTable(max_am, max_spin)
    for comb in occupy(shell, l, e_num):
        am, spin = calc_vals(comb)
        if am < 0 or spin < 0:
            continue
        t.increment(am, spin)

    return t