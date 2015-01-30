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
    am = 0
    spin = 0
    for orb in orbs:
        am += orb.ml
        spin += orb.spin

    return am, spin


class TermSymbol:
    def __init__(self, mult, am):
        self.am = abs(am)
        self.mult = mult

    def __str__(self):
        symbols = 'SPDFGHKLMN'
        return '{}{}'.format(self.mult, symbols[abs(self.am)])

    def __repr__(self):
        return str(self)

    def __eq__(self, o):
        if self.am == o.am and self.mult == o.mult:
            return True
        return False


def find_term_symbol(orbs, latex=False):
    """Generate the term symbol for the given set of orbitals
    WARNING: It does not check if the set of orbitals is valid
    """
    am, spin = calc_vals(orbs)
    return TermSymbol(2*spin + 1, am)


class TermTable:
    def __init__(self, max_mult, max_am):
        self.max_am = max_am
        self.width = max_am + 1
        self.max_mult = max_mult
        self.min_mult = max_mult % 2
        self.height = (max_mult + 1) // 2
        self.table = np.zeros((self.height, self.width), dtype=np.dtype(int))
        # TODO: add combinations to each term
        self.combs = {}

    def __str__(self):
        """Flip the table for printing"""
        return str(self.table[::-1])

    def __eq__(self, o):
        if self.max_am != o.max_am or self.max_mult != o.max_mult:
            return False
        if not (self.table == o.table).all():
            return False
        return True

    def get(self, mult, am):
        return self.table[(mult - 1) // 2][am]

    def set(self, mult, am, val):
        self.table[(mult - 1) // 2][am] = val

    def increment(self, mult, am):
        self.table[(mult - 1) // 2][am] += 1

    def clean_table(self):
        """Remove all the lower manifestations of terms
        i.e. subtract the value of (a,b) from other terms where x<=a, y<=b"""
        for i in reversed(range(len(self.table))):
            for j in reversed(range(len(self.table[0]))):
                count = self.table[i, j]
                self.table[:i+1, :j+1] -= count
                self.table[i, j] = count

    def print(self, style='table'):
        """Print the table in a nice format"""
        out = ''
        if style == 'table':
            line = '{}' + '{:<4}'*len(self.table[0])
            for i, row in enumerate(self.table):
                out += line.format(i, *row)

        print(out)


def subshell_terms(shell, l, e_num):
    max_am = l*e_num
    if e_num <= 2*l + 1:
        max_mult = e_num + 1
    else:
        max_mult = 4*l + 3 - e_num
    t = TermTable(max_mult, max_am)
    for comb in occupy(shell, l, e_num):
        am, spin = calc_vals(comb)
        if am < 0 or spin < 0:
            continue
        t.increment(int(2*spin + 1), am)

    return t