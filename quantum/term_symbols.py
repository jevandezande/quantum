from itertools import combinations, product
from fractions import Fraction as Frac
import numpy as np
from copy import deepcopy

AM_SYMBOLS = 'spdfghklmn'
AM_SYMBOLS_UP = 'SPDFGHKLMN'


class SpinOrbital:
    def __init__(self, n, l, ml, spin):
        """A spin orbital"""
        if not isinstance(n, int) or n < 1:
            raise SyntaxError("Shells are integers starting at 1")

        if isinstance(l, str):
            try:
                l = AM_SYMBOLS.index(l)
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
        orb = AM_SYMBOLS[self.l]
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
        if not TermSymbol.check(mult, am):
            raise SyntaxError("Multiplicity and angular momentum must be ints")
        self.am = abs(am)
        self.mult = mult

    def __str__(self):
        return '{}{}'.format(self.mult, AM_SYMBOLS_UP[abs(self.am)])

    def __repr__(self):
        return str(self)

    def __eq__(self, o):
        if self.am == o.am and self.mult == o.mult:
            return True
        return False

    @staticmethod
    def check(mult, am):
        if not isinstance(mult, int) or mult < 1 or not isinstance(am, int):
            return False
        return True

    @staticmethod
    def latex(mult, am):
        """Return a latex based representation of the term symbol"""
        if not TermSymbol.check(mult, am):
            raise SyntaxError("Multiplicity and angular momentum must be ints")
        return "$^{}${}".format(mult, AM_SYMBOLS_UP[am])


def find_term_symbol(orbs, latex=False):
    """Generate the term symbol for the given set of orbitals
    WARNING: It does not check if the set of orbitals is valid
    """
    am, spin = calc_vals(orbs)
    return TermSymbol(int(2*spin + 1), am)


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

    def __mul__(self, o):
        """Multiple two term tables to get the product table

        WARNING: Do not multiply term tables coming from the same subshell

        TODO: Fill out table
        """
        max_mult = self.max_mult + o.max_mult - 1
        max_am = self.max_am + o.max_am
        t = TermTable(max_mult, max_am)

        return t

    def get(self, mult, am):
        return self.table[(mult - 1) // 2][am]

    def set(self, mult, am, val):
        self.table[(mult - 1) // 2][am] = val

    def increment(self, mult, am):
        self.table[(mult - 1) // 2][am] += 1

    def cleaned(self):
        """Create a new TermTable with all lower manifestations of terms removed
        i.e. subtract the value of (a,b) from other terms where x<=a, y<=b"""
        cleaned = deepcopy(self)
        for i in reversed(range(len(cleaned.table))):
            for j in reversed(range(len(cleaned.table[0]))):
                count = cleaned.table[i, j]
                cleaned.table[:i+1, :j+1] -= count
                cleaned.table[i, j] = count

        return cleaned

    def string(self, style='table'):
        """Print the table in a nice format"""
        out = ''
        if style == 'table':
            top_line = 'M\\L|' + ' {:> 3}'*self.width + '\n'
            out += top_line.format(*list(range(self.width)))
            out += '-'*(4 + 4*self.width) + '\n'
            line = '{:> 3}|' + ' {:> 3}'*self.width + '\n'
            for i, row in reversed(list(enumerate(self.table))):
                mult = self.min_mult + i*2 + 1
                out += line.format(mult, *row)
                out += '-'*(4 + 4*self.width) + '\n'

        elif style == 'latex':
            out += '\\begin{tabular}{ r |' + ' c'*self.width + ' } \n'
            top_line = 'M\\L ' + '& {:> 6} '*self.width + '\\hl \n'
            out += top_line.format(*list(range(self.width)))
            line = '{:>3} ' + '& {:>6} '*self.width + '\\\\ \n'
            for i, row in reversed(list(enumerate(self.table))):
                mult = self.min_mult + i*2 + 1
                symbols = [TermSymbol.latex(mult, am) for am in range(len(row))]
                out += line.format(mult, *symbols)
            out += '\\end{tabular}'

        elif style == 'latex-crossed':
            out += '\\begin{tabular}{ r |' + ' c'*self.width + ' } \n'
            top_line = 'M\\L ' + '& {:> 10} '*self.width + '\\hl \n'
            out += top_line.format(*list(range(self.width)))
            t_form = '& {:>10} '
            for i, row in reversed(list(enumerate(self.table))):
                mult = self.min_mult + i*2 + 1
                out += '{:>3} '.format(mult)
                for am in range(len(row)):
                    symb = TermSymbol.latex(mult, am)
                    count = self.table[i, am]
                    t = TermSymbol.latex(mult, am)
                    if count == 0:
                        out += t_form.format('\\x{' + t + '}')
                    else:
                        out += t_form.format('\\' + 'O'*count + '{' + t + '}')
                out += '\\\\ \n'
            out += '\\end{tabular}'

        return out


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


def multiple_subshell_terms(*subshells):
    """Subshells is a list of tuples of the form
    (shell, l, e_num)
    """

    if len(subshells) == 0:
        return TermTable(0, 0)

    occupied = []
    max_am = 0
    max_mult = -len(subshells) + 1
    for shell, l, e_num in subshells:
        max_am += l * e_num
        if e_num <= 2*l + 1:
            max_mult += e_num + 1
        else:
            max_mult += 4*l + 3 - e_num
        occupied.append(list(occupy(shell, l, e_num)))

    t = TermTable(max_mult, max_am)
    for comb in product(*occupied):
        am = 0
        spin = 0
        for subshell in comb:
            am_s, spin_s = calc_vals(subshell)
            am += am_s
            spin += spin_s

        if am < 0 or spin < 0:
            continue
        t.increment(int(2*spin + 1), am)

    return t

