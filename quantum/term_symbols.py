from itertools import combinations, product
from fractions import Fraction as Frac
from copy import deepcopy
from abc import ABCMeta, abstractmethod

import numpy as np


ATOMIC_AM_SYMBOLS = 'spdfghiklmnoqrtuvwxyz'
ATOMIC_AM_SYMBOLS_UP = 'SPDFGHIKLMNOQRTUVWXYZ'
DIATOMIC_AM_SYMBOLS = 'σπδφγηικμνo'
DIATOMIC_AM_SYMBOLS_UP = 'ΣΠ∆ΦΓHIKMNO'


class Orbital:
    def __init__(self, n, l, ml, spin):
        """An orbital
        :param n: shell
        :param l: angular momentum
        :param ml: directional angular momentum (azimuthal, etc.)
        :param spin: electron spin
        """
        if not isinstance(n, int) or n < 1:
            raise SyntaxError("Shells are integers starting at 1")

        if not isinstance(l, int) or l < 0:
            raise SyntaxError("Orbitals are ints >=0 or the appropriate string")

        if ml > l or not isinstance(ml, int):
            raise SyntaxError("Angular momentum (ml) is a positive " +
                              "integer such that -l <= ml <= l")

        if spin == 1 or spin == Frac(1 / 2) or spin == 'alpha':
            self.spin = Frac(1 / 2)
        elif spin == -1 or spin == Frac(-1 / 2) or spin == 'beta':
            self.spin = Frac(-1 / 2)
        else:
            raise SyntaxError("Invalid spin")

        self.n = n
        self.l = l
        self.ml = ml

    def __str__(self):
        spin = 'a' if self.spin > 0 else 'b'
        return '{n}{l}_{{{ml}}}{spin}'.format(n=self.n, l=self.orb_symbol,
                                              ml=self.ml, spin=spin)

    def __eq__(self, o):
        if type(self) == type(o) \
                and self.n == o.n and self.l == o.l \
                and self.ml == o.ml and self.spin == o.spin:
            return True
        return False

    def __repr__(self):
        return self.__str__()


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
            raise SyntaxError("Diatomic orbitals may only have ml = +- l")
        self.orb_symbol = DIATOMIC_AM_SYMBOLS[l]


def spin_iterator():
    """Iterate through alpha (1/2) and beta (-1/2) spins"""
    yield Frac(1, 2)
    yield -Frac(1, 2)


def atomic_spinorbitals_iterator(shell, l):
    """Makes an iterator over all SpinOrbitals in the specified subshell
    :param shell: orbital shell
    :param l: angular momentum of the desired subshell
    """
    for ml, spin in product(range(l, -l - 1, -1), spin_iterator()):
        yield AtomicSpinOrbital(n=shell, l=l, ml=ml, spin=spin)


def diatomic_spinorbitals_iterator(shell, l):
    """Makes an iterator over all DiatomicSpinOrbitals in specified subshell
    :param shell: orbital shell
    :param l: angular momentum of the desired subshell
    """
    mls = [l, -l] if l > 0 else [0]
    for ml, spin in product(mls, spin_iterator()):
        yield DiatomicSpinOrbital(n=shell, l=l, ml=ml, spin=spin)


def occupy(iterator, e_num):
    """Make an iterator of all possible combinations
    :param iterator: an orbital iterator
    :param e_num: number of electrons
    """
    return combinations(iterator, e_num)


def calc_vals(orbs):
    """Calculate the total angular momentum and spin
    :param orbs: an iterator containing Orbitals
    """
    am = 0
    spin = 0
    for orb in orbs:
        am += orb.ml
        spin += orb.spin

    return am, spin


class TermSymbol:
    """Quantum term symbol class for atoms"""

    def __init__(self, mult, am, orbital_type='atomic'):
        """
        :param mult: multiplicity of the term symbol
        :param am: angular momentum of the term symbol
        :param orbital_type: the type of orbitals that are used (i.e. atomic or
            diatomic)
        """
        if not TermSymbol.check(mult, am):
            raise SyntaxError("Multiplicity and angular momentum must be ints")
        self.am = abs(am)
        self.mult = mult
        if orbital_type == 'atomic':
            self.am_symbols = ATOMIC_AM_SYMBOLS_UP
        elif orbital_type == 'diatomic':
            self.am_symbols = DIATOMIC_AM_SYMBOLS_UP
        else:
            raise SyntaxError("Only atomic and diatomic orbitals are currently"
                              "supported")

    def __str__(self):
        return '{}{}'.format(self.mult, self.am_symbols[abs(self.am)])

    def __repr__(self):
        return str(self)

    def __eq__(self, o):
        if self.am == o.am and self.mult == o.mult:
            return True
        return False

    @staticmethod
    def check(mult, am):
        """Check if the multiplicity and angular momentum are valid"""
        if not isinstance(mult, int) or mult < 1 or not isinstance(am, int):
            return False
        return True

    @staticmethod
    def latex(mult, am, orbital_type):
        """Return a latex based representation of the term symbol
        :param mult: multiplicity of the term symbol
        :param am: angular momentum of the term symbol
        """
        if orbital_type == 'atomic':
            am_symbols = ATOMIC_AM_SYMBOLS_UP
        elif orbital_type == 'diatomic':
            am_symbols = DIATOMIC_AM_SYMBOLS_UP
        if not TermSymbol.check(mult, am):
            raise SyntaxError("Multiplicity and angular momentum must be ints")
        return "$^{}${}".format(mult, am_symbols[am])

    @staticmethod
    def table(min_mult, max_mult, min_am, max_am, orbital_type):
        """
        :param orbital_type: the type of orbitals that are used (i.e. atomic or
                                                                 diatomic)
        """
        height = int((max_mult - min_mult) / 2)
        width = int(max_am - min_am)
        print(width, height)
        terms = [[''] * width for _ in range(height)]
        for i, am in enumerate(range(min_am, max_am)):
            for j, mult in enumerate(range(min_mult, max_mult, 2)):
                terms[j][i] = TermSymbol(mult, am, orbital_type)

        return terms


def find_term_symbol(orbs):
    """Generate the term symbol for the given set of orbitals
    :param orbs: an iterable containing SpinOrbitals
    WARNING: It does not check if the set of orbitals is valid
    """
    am, spin = calc_vals(orbs)
    return TermSymbol(int(2 * spin + 1), am)


class TermTable:
    def __init__(self, max_mult, max_am):
        """Set up the table

        :param max_mult: maximum possible multiplicity
        :param max_am: maximum possible angular momentum
        """
        self.max_am = max_am
        self.width = max_am + 1
        self.max_mult = max_mult
        self.min_mult = (max_mult + 1) % 2 + 1
        self.height = (max_mult + 1) // 2
        self.table = np.zeros((self.height, self.width), dtype=np.dtype(int))
        self.orbital_type = None

    def __str__(self):
        """Flips the table for printing in standard form"""
        return str(self.table[::-1])

    def __eq__(self, o):
        if self.max_am != o.max_am or self.max_mult != o.max_mult:
            return False
        if not (self.table == o.table).all():
            return False
        return True

    def __mul__(self, o):
        """Multiple two term tables to get the product table

        WARNING: Do not multiply TermTables coming from the same subshell
        WARNING: Do not multiply TermTables that have not been cleaned

        TODO: Fill out table
        """
        if not isinstance(o, TermTable) or type(self) != type(o):
            raise SyntaxError("Can only multiply a TermTable by a TermTable")
        max_mult = self.max_mult + o.max_mult - 1
        max_am = self.max_am + o.max_am
        t = TermTable(max_mult, max_am)

        for i, row in enumerate(self.table):
            for j, count in enumerate(row):
                mult = self.min_mult + 2 * i
                am = j
                for k, o_row in enumerate(o.table):
                    for l, o_count in enumerate(o_row):
                        o_mult = o.min_mult + 2 * i
                        o_am = l
                        t.add(mult + o_mult - 1, am + o_am, count * o_count)

                        if mult > 1 and o_mult > 1 and mult + o_mult > 3:
                            t.add(abs(mult - o_mult + 1), am + o_am,
                                  count * o_count)
                        if am > 0 and o_am > 0:
                            t.add(mult + o_mult - 1, abs(am - o_am),
                                  count * o_count)
                            if mult > 1 and o_mult > 1 and mult + o_mult > 3:
                                t.add(abs(mult - o_mult + 1), abs(am - o_am),
                                      count * o_count)

        return t

    __rmul__ = __mul__

    @abstractmethod
    def cleaned(self):
        pass

    def get(self, mult, am):
        """Get the value at the specified mult and am"""
        return self.table[(mult - 1) // 2][am]

    def set(self, mult, am, val):
        """Set the value at the specified mult and am"""
        self.table[(mult - 1) // 2][am] = val

    def add(self, mult, am, val):
        """Set the value at the specified mult and am"""
        self.table[(mult - 1) // 2][am] += val

    def increment(self, mult, am):
        """Increment the value at the specified mult and am"""
        self.table[(mult - 1) // 2][am] += 1

    def string(self, style='table'):
        """Print the table in a nice format
        :param style: the style the table should be output in
                        table, latex, latex-crossed, latex-table
        """
        out = ''
        if style == 'table':
            top_line = 'M\\L|' + ' {:> 3}' * self.width + '\n'
            out += top_line.format(*list(range(self.width)))
            out += '-' * (4 + 4 * self.width) + '\n'
            line = '{:> 3}|' + ' {:> 3}' * self.width + '\n'
            for i, row in reversed(list(enumerate(self.table))):
                mult = self.min_mult + i * 2
                out += line.format(mult, *row)
                out += '-' * (4 + 4 * self.width) + '\n'

        elif style == 'latex':
            out += '\\begin{tabular}{ r |' + ' c' * self.width + ' } \n'
            top_line = 'M\\L ' + '& {:> 6} ' * self.width + '\\hl \n'
            out += top_line.format(*list(range(self.width)))
            line = '{:>3} ' + '& {:>6} ' * self.width + '\\\\ \n'
            for i, row in reversed(list(enumerate(self.table))):
                mult = self.min_mult + i * 2
                symbols = [TermSymbol.latex(mult, am, self.orbital_type) for am
                           in range(len(row))]
                out += line.format(mult, *symbols)
            out += '\\end{tabular}'

        elif style == 'latex-crossed':
            out += '\\begin{tabular}{ r |' + ' c' * self.width + ' } \n'
            top_line = 'M\\L ' + '& {:> 10} ' * self.width + '\\hl \n'
            out += top_line.format(*list(range(self.width)))
            t_form = '& {:>10} '
            for i, row in reversed(list(enumerate(self.table))):
                mult = self.min_mult + i * 2
                out += '{:>3} '.format(mult)
                for am in range(len(row)):
                    count = self.table[i, am]
                    t = TermSymbol.latex(mult, am, self.orbital_type)
                    if count == 0:
                        out += t_form.format('\\x{' + t + '}')
                    else:
                        out += t_form.format('\\' + 'O' * count + '{' + t + '}')
                out += '\\\\ \n'
            out += '\\end{tabular}'

        elif style == 'latex-table':
            out += '\\begin{tabular}{ r |' + ' c' * self.width + ' } \n'
            top_line = 'M\\L ' + '& {:>3} ' * self.width + '\\hl \n'
            out += top_line.format(*list(range(self.width)))
            line = '{:>3} ' + '& {:>3} ' * self.width + '\\\\ \n'
            for i, row in reversed(list(enumerate(self.table))):
                mult = self.min_mult + i * 2
                out += line.format(mult, *row)
            out += '\\end{tabular}'

        return out


class AtomicTermTable(TermTable):
    """A table that contains the number of terms at a specified multiplicity and
    angular momentum.

    Can remove all manifestations of terms at lower multiplicity or angular
    momentum. Thus producing the standard scorecard.
    """

    def __init__(self, max_mult, max_am):
        super().__init__(max_mult, max_am)
        self.orbital_type = 'atomic'

    def cleaned(self):
        """Create a new AtomicTermTable with all lower manifestations of terms
        removed.
        i.e. subtract the value of x=mult, y=am from terms where x<mult, y<am
        :returns AtomicTermTable:
        """
        cleaned = deepcopy(self)
        for i in reversed(range(len(cleaned.table))):
            for j in reversed(range(len(cleaned.table[0]))):
                count = cleaned.table[i, j]
                cleaned.table[:i + 1, :j + 1] -= count
                cleaned.table[i, j] = count

        return cleaned


class DiatomicTermTable(TermTable):
    """A diatomic version of TermTable"""

    def __init__(self, max_mult, max_am):
        super().__init__(max_mult, max_am)
        self.orbital_type = 'diatomic'

    def cleaned(self):
        """Create a new DiatomicTermTable with all lower manifestations of terms
        removed.
        i.e. subtract the value of x=mult, y=am from terms where x<mult, y=am
        :returns DiatomicTermTable:
        """
        cleaned = deepcopy(self)
        for i in reversed(range(len(cleaned.table))):
            for j in reversed(range(len(cleaned.table[0]))):
                count = cleaned.table[i, j]
                cleaned.table[:i + 1, j] -= count
                cleaned.table[i, j] = count

        return cleaned


def subshell_terms(orbital_type, shell, l, e_num):
    """Iterate over all possible combinations of electrons in orbitals
    :param orbital_type: type of orbitals desired
    :param shell: orbital shell
    :param l: orbital angular momentum
    :param e_num: number of electrons
    :returns: TermTable of corresponding to orbital_type
    """
    max_am = l * e_num
    if e_num <= 2 * l + 1:
        max_mult = e_num + 1
    else:
        max_mult = 4 * l + 3 - e_num

    if orbital_type == 'atomic':
        iterator = atomic_spinorbitals_iterator(shell, l)
        t = AtomicTermTable(max_mult, max_am)
    elif orbital_type == 'diatomic':
        iterator = diatomic_spinorbitals_iterator(shell, l)
        t = DiatomicTermTable(max_mult, max_am)
    else:
        raise SyntaxError("Invalid orbital type.")

    for comb in occupy(iterator, e_num):
        am, spin = calc_vals(comb)
        if am < 0 or spin < 0:
            continue
        t.increment(int(2 * spin + 1), am)

    return t


def multiple_subshell_terms(*subshells):
    """Iterate over all possible combinations of electrons in orbitals
    Currently only for atomic orbitals
    :param subshells:  an iterable where each term is (shell, l, e_num)
    :returns: AtomicTermTable
    """

    if len(subshells) == 0:
        return AtomicTermTable(0, 0)

    occupied = []
    max_am = 0
    max_mult = -len(subshells) + 1
    for shell, l, e_num in subshells:
        max_am += l * e_num
        if e_num <= 2 * l + 1:
            max_mult += e_num + 1
        else:
            max_mult += 4 * l + 3 - e_num
        iterator = atomic_spinorbitals_iterator(shell, l)
        occupied.append(list(occupy(iterator, e_num)))

    t = AtomicTermTable(max_mult, max_am)
    for comb in product(*occupied):
        am = 0
        spin = 0
        for subshell in comb:
            am_s, spin_s = calc_vals(subshell)
            am += am_s
            spin += spin_s

        if am < 0 or spin < 0:
            continue
        t.increment(int(2 * spin + 1), am)

    return t


def all_atomic_term_tables(max_am):
    """Iterate through all the AtomicTermTables up to a specified angular momentum

    Since particle-hole equivalence makes the term symbol tables symmetric
    around the point where the number of electrons equals 2*am +1
    e.g. d^2 is equivalent to d^8
    :param max_am: maximum desired angular momentum subshell
    """

    for am in range(max_am):
        for e_num in range(1, 2 * am + 2):
            yield subshell_terms('atomic', am + 1, am, e_num)
