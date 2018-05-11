from itertools import combinations, product
from fractions import Fraction as Frac
from copy import deepcopy
from abc import ABCMeta, abstractmethod

import numpy as np

from quantum.orbitals import *


def spin_iterator():
    """Iterate through alpha (1/2) and beta (-1/2) spins"""
    yield Frac(1, 2)
    yield -Frac(1, 2)


def atomic_spinorbitals_iterator(shell, l):
    """
    Makes an iterator over all SpinOrbitals in the specified subshell
    :param shell: orbital shell
    :param l: angular momentum of the desired subshell
    :yields: all possible AtomicSpinOrbitals
    """
    for ml, spin in product(range(l, -l - 1, -1), spin_iterator()):
        yield AtomicSpinOrbital(n=shell, l=l, ml=ml, spin=spin)


def diatomic_spinorbitals_iterator(shell, l):
    """
    Makes an iterator over all DiatomicSpinOrbitals in specified subshell
    :param shell: orbital shell
    :param l: angular momentum of the desired subshell
    :yields: all possible DiatomicSpinOrbitals
    """
    mls = [l, -l] if l > 0 else [0]
    for ml, spin in product(mls, spin_iterator()):
        yield DiatomicSpinOrbital(n=shell, l=l, ml=ml, spin=spin)


def occupy(iterator, e_num):
    """
    Make an iterator of all possible combinations
    :param iterator: an orbital iterator
    :param e_num: number of electrons
    :returns: all possible combinations
    """
    yield from combinations(iterator, e_num)


def calc_vals(orbs):
    """
    Calculate the total angular momentum and spin
    :param orbs: an iterator containing Orbitals
    :returns: (angular momentum, spin)
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
        :param orbital_type: the type of orbitals that are used (i.e. atomic or diatomic)
        """
        if not TermSymbol.check(mult, am):
            raise SyntaxError("Multiplicity must be greater than 0 and "
                              "multiplicity and angular momentum must be ints.")
        self.am = abs(am)
        self.mult = mult
        if orbital_type == 'atomic':
            self.am_symbols = ATOMIC_AM_SYMBOLS_UP
        elif orbital_type == 'diatomic':
            self.am_symbols = DIATOMIC_AM_SYMBOLS_UP
        else:
            raise SyntaxError("Only atomic and diatomic orbitals are currently supported.")
        self.orbital_type = orbital_type

    def __str__(self):
        return f'{self.mult}{self.am_symbol}'

    def __repr__(self):
        return f"<{self.orbital_type.title()}TermSymbol {self}>"

    def __eq__(self, o):
        if self.am == o.am and self.mult == o.mult:
            return True
        return False

    @property
    def am_symbol(self):
        return self.am_symbols[abs(self.am)]

    @staticmethod
    def check(mult, am):
        """
        Check if the multiplicity and angular momentum are valid
        :param mult: multiplicity
        :param am: angular momentum
        :returns: bool of validity
        """
        if mult < 1 or not isinstance(am, int):
            return False
        if not isinstance(mult, int) and not (isinstance(mult, Frac) and mult.denominator == 1):
            return False
        return True

    @staticmethod
    def latex(mult, am, orbital_type):
        """
        Generate a latex based representation of the term symbol
        :param mult: multiplicity of the term symbol
        :param am: angular momentum of the term symbol
        :returns: string of LaTeX representation
        """
        if orbital_type == 'atomic':
            am_symbols = ATOMIC_AM_SYMBOLS_UP
        elif orbital_type == 'diatomic':
            am_symbols = DIATOMIC_AM_SYMBOLS_UP
        if not TermSymbol.check(mult, am):
            raise SyntaxError("Multiplicity and angular momentum must be ints.")
        return f'$^{mult}${am_symbols[am]}'

    @staticmethod
    def table(min_mult, max_mult, min_am, max_am, orbital_type):
        """
        Generate a table of TermSymbols
        :param orbital_type: the type of orbitals that are used (i.e. atomic or diatomic)
        :returns: 2D list of TermSymbols
        """
        height = int((max_mult - min_mult) / 2)
        width = int(max_am - min_am)
        terms = [[''] * width for _ in range(height)]
        for i, am in enumerate(range(min_am, max_am)):
            for j, mult in enumerate(range(min_mult, max_mult, 2)):
                terms[j][i] = TermSymbol(mult, am, orbital_type)

        return terms

    def num_microstates(self):
        """
        Number of microstates for the TermSymbol

        e.g. ^3D

        m\l |-D -P | S  P  D |      s\l |-D -P | S  P  D |
        ----------------------      ----------------------
        3   | 1  1 | 1  1  1 |      1   | 1  1 | 1  1  1 |
        1   | 1  1 | 1  1  1 |  ==  0   | 1  1 | 1  1  1 |
        ----------------------      ----------------------
        3   | 1  1 | 1  1  1 |     -1   | 1  1 | 1  1  1 |
        ----------------------      ----------------------

        e.g. ^4F

        | m\l |-F -D -P | S  P  D |     | s\l |-F -D -P | S  P  D  F |
        ---------------------------     ------------------------------
        |  4  | 1  1  1 | 1  1  1 |     | 3/2 | 1  1  1 | 1  1  1  1 |
        |  2  | 1  1  1 | 1  1  1 |     | 1/2 | 1  1  1 | 1  1  1  1 |
        ---------------------------     ------------------------------
        |  2  | 1  1  1 | 1  1  1 |     |-1/2 | 1  1  1 | 1  1  1  1 |
        |  4  | 1  1  1 | 1  1  1 |     |-3/2 | 1  1  1 | 1  1  1  1 |
        ---------------------------     ------------------------------
        """
        return (2*self.am + 1) * self.mult

    def form_jstates(self):
        """
        Generates the SOTermSymbol states that can arise from the given TermSymbol
        :yields: all jstate SOTermSymbols
        """
        s = Frac(self.mult - 1, 2)
        max_j = self.am + s
        # Cannot go below 0 for j value
        min_j = max(self.am - s, s%1)
        span = max_j - min_j
        for j in min_j + np.arange(span + 1):
            yield SOTermSymbol(self.mult, self.am, j)


class SOTermSymbol(TermSymbol):

    def __init__(self, mult, am, j, orbital_type='atomic'):
        """
        :param mult: multiplicity of the term symbol
        :param am: angular momentum of the term symbol
        :param j: total angular momentum quantum number, |l + s|
        :param orbital_type: the type of orbitals that are used (i.e. atomic or
            diatomic)
        """
        super().__init__(mult, am, orbital_type)
        if not isinstance(j, (int, Frac)):
            raise SyntaxError("Invalid j")
        self.j = j

    def __str__(self):
        return f'{mult}{self.am_symbol}_{self.j}'

    def __repr__(self):
        return f'<{self.orbital_type.title()}SOTermSymbol {self}>'

    def __eq__(self, o):
        if not isinstance(o, SOTermSymbol):
            raise SyntaxError(f'Cannot compare SOTermSymbol and {type(o)}.')
        if self.am == o.am and self.mult == o.mult and self.j == o.j:
            return True
        return False


def find_term_symbol(orbs, j=False):
    """
    Generate the term symbol for the given set of orbitals
    WARNING: It does not check if the set of orbitals is valid
    :param orbs: an iterable containing SpinOrbitals
    :returns: TermSymbol
    """
    am, spin = calc_vals(orbs)
    if not j:
        return TermSymbol(int(2 * spin + 1), am)
    else:
        """TODO: Confirm this is the best way to generate J"""
        return SOTermSymbol(int(2 * abs(spin) + 1), am, abs(am + spin))


class TermTable:
    def __init__(self, max_mult, max_am, clean=False):
        """
        Set up the TermTable

        :param max_mult: maximum multiplicity of the table (height)
        :param max_am: maximum angular momentum of the table (width)
        :param clean: boolean describing if microstates have been removed
        """
        self.max_am = max_am
        self.width = max_am + 1
        self.max_mult = max_mult
        self.min_mult = (max_mult + 1) % 2 + 1
        self.height = (max_mult + 1) // 2
        self.table = np.zeros((self.height, self.width), dtype=np.dtype(int))
        self.orbital_type = None
        self._clean = clean

    def __repr__(self):
        return f'TermTable({self.table.tolist()})'

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
        """
        Multiple two term tables to get the product table

        WARNING: Do not multiply TermTables coming from the same subshell
        WARNING: Do not multiply TermTables that have not been cleaned

        TODO: Fill out table
        """
        if not isinstance(o, TermTable) or type(self) != type(o):
            raise SyntaxError("Can only multiply a TermTable by a TermTable.")
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
        """
        Print the table in a nice format
        :param style: the style the table should be output in table, latex,
                      latex-crossed, latex-table
        :returns: string in specified form
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
    """
    A table that contains the number of terms at a specified multiplicity and
    angular momentum.

    Can remove all manifestations of terms at lower multiplicity or angular
    momentum. Thus producing the standard scorecard.
    """

    def __init__(self, max_mult, max_am, clean=False):
        """
        :param max_mult: maximum multiplicity of the table (height)
        :param max_am: maximum angular momentum of the table (width)
        """
        super().__init__(max_mult, max_am, clean)
        self.orbital_type = 'atomic'

    def cleaned(self):
        """
        Create a new AtomicTermTable with all lower manifestations of terms
        removed.
        i.e. subtract the value of x=mult, y=am from terms where x<mult, y<am
        :returns: AtomicTermTable
        """
        cleaned = deepcopy(self)
        for i in reversed(range(self.height)):
            for j in reversed(range(self.width)):
                count = cleaned.table[i, j]
                cleaned.table[:i + 1, :j + 1] -= count
                cleaned.table[i, j] = count
        cleaned._clean = True

        return cleaned


class DiatomicTermTable(TermTable):
    """A diatomic version of TermTable"""

    def __init__(self, max_mult, max_am, clean=False):
        """
        :param max_mult: maximum multiplicity of the table (height)
        :param max_am: maximum angular momentum of the table (width)
        """
        super().__init__(max_mult, max_am, clean=False)
        self.orbital_type = 'diatomic'

    def cleaned(self):
        """
        Create a new DiatomicTermTable with all lower manifestations of terms
        removed.
        i.e. subtract the value of x=mult, y=am from terms where x<mult, y=am
        :returns: DiatomicTermTable
        """
        cleaned = deepcopy(self)
        for i in reversed(range(self.height)):
            for j in reversed(range(self.width)):
                count = cleaned.table[i, j]
                cleaned.table[:i + 1, j] -= count
                cleaned.table[i, j] = count
        cleaned._clean = True

        return cleaned


def subshell_terms(orbital_type, shell, l, e_num):
    """
    Iterate over all possible combinations of electrons in orbitals.
    :param orbital_type: type of orbitals desired
    :param shell: orbital shell
    :param l: orbital angular momentum
    :param e_num: number of electrons
    :returns: TermTable corresponding to orbital_type
    """
    a, b = int(np.ceil(e_num/2)), int(np.floor(e_num / 2))
    max_am = sum(range(l - a + 1, l + 1)) + sum(range(l - b + 1, l + 1))
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
        t.increment(int(2 * abs(spin) + 1), abs(am))

    return t


def multiple_subshell_terms(orbital_type, *subshells):
    """
    Iterate over all possible combinations of electrons in orbitals.
    Currently only for atomic orbitals
    :param subshells:  an iterable where each term is (shell, l, e_num)
    :returns: TermTable
    """
    if len(subshells) == 0:
        raise SyntaxError("Need subshells.")

    if orbital_type not in ['atomic', 'diatomic']:
        SyntaxError("Invalid orbital type.")

    occupied = []
    max_am = 0
    max_mult = 1
    for shell, l, e_num in subshells:
        if orbital_type == 'atomic':
            max_occ = 4 * l + 2
            iterator = atomic_spinorbitals_iterator(shell, l)
        elif orbital_type == 'diatomic':
            max_occ = 4 if l > 0 else 2
            iterator = diatomic_spinorbitals_iterator(shell, l)
        max_mult += min(e_num, max_occ - e_num)
        max_am += l * min(e_num, max_occ - e_num)
        occupied.append(list(occupy(iterator, e_num)))

    if orbital_type == 'atomic':
        t = AtomicTermTable(max_mult, max_am)
    elif orbital_type == 'diatomic':
        t = DiatomicTermTable(max_mult, max_am)

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
    """
    Iterate through all the AtomicTermTables up to a specified angular momentum

    Since particle-hole equivalence makes the term symbol tables symmetric
    around the point where the number of electrons equals 2*am +1
    e.g. d^2 is equivalent to d^8
    :param max_am: maximum desired angular momentum subshell
    :yields: TermTable
    """

    for am in range(max_am):
        for e_num in range(1, 2 * am + 2):
            yield subshell_terms('atomic', am + 1, am, e_num)
