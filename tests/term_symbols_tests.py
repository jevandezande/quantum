from fractions import Fraction

from nose.tools import *
from numpy.testing import assert_equal as np_assert_equal

from quantum.term_symbols import *
from quantum.orbitals import *


def setup():
    pass


def teardown():
    pass


def test_spin_iterator():
    a = Fraction(1, 2)
    b = Fraction(-1, 2)
    assert_equal(list(spin_iterator()), [a, b])


def test_atomic_spin_orbitals_iterator():
    p_orbs = ['2p_{1}a', '2p_{1}b', '2p_{0}a', '2p_{0}b', '2p_{-1}a',
              '2p_{-1}b']
    p_l = list(map(str, atomic_spinorbitals_iterator(2, 1)))
    assert_equal(p_l, p_orbs)


def test_diatomic_spin_orbitals_iterator():
    pi_orbs = ['2π_{1}a', '2π_{1}b', '2π_{-1}a', '2π_{-1}b']
    pi_l = list(map(str, diatomic_spinorbitals_iterator(2, 1)))
    assert_equal(pi_l, pi_orbs)


def test_occupy():
    iterator1 = atomic_spinorbitals_iterator(1, 0)
    assert_equal(list(occupy(iterator1, 0)), [()])
    iterator2 = atomic_spinorbitals_iterator(2, 1)
    assert_equal(list(occupy(iterator2, 0)), [()])

    iterator3 = atomic_spinorbitals_iterator(1, 0)
    occ3 = list(occupy(iterator3, 1))
    assert_equal(occ3[0][0], AtomicSpinOrbital(n=1, l=0, ml=0, spin='alpha'))
    assert_equal(len(occ3), 2)

    iterator4 = atomic_spinorbitals_iterator(1, 0)
    occ4 = list(occupy(iterator4, 2))
    assert_equal(occ4[0][0], AtomicSpinOrbital(n=1, l=0, ml=0, spin='alpha'))
    assert_equal(len(occ4), 1)

    iterator5 = atomic_spinorbitals_iterator(2, 1)
    occ5 = list(occupy(iterator5, 2))
    assert_equal(len(occ5), 15)

    iterator6 = diatomic_spinorbitals_iterator(1, 0)
    occ6 = list(occupy(iterator6, 1))
    assert_equal(occ6[0][0], DiatomicSpinOrbital(n=1, l=0, ml=0, spin='alpha'))
    assert_equal(occ6[1][0], DiatomicSpinOrbital(n=1, l=0, ml=0, spin='beta'))
    assert_equal(len(occ6), 2)

    iterator7 = diatomic_spinorbitals_iterator(1, 1)
    occ7 = list(occupy(iterator7, 1))
    assert_equal(occ7[0][0], DiatomicSpinOrbital(n=1, l=1, ml=1, spin='alpha'))
    assert_equal(occ7[1][0], DiatomicSpinOrbital(n=1, l=1, ml=1, spin='beta'))
    assert_equal(len(occ7), 4)

    iterator7 = diatomic_spinorbitals_iterator(1, 1)
    occ7 = list(occupy(iterator7, 1))
    assert_equal(occ7[0][0], DiatomicSpinOrbital(n=1, l=1, ml=1, spin='alpha'))
    assert_equal(occ7[3][0], DiatomicSpinOrbital(n=1, l=1, ml=-1, spin='beta'))
    assert_equal(len(occ7), 4)

    iterator8 = diatomic_spinorbitals_iterator(1, 3)
    occ8 = list(occupy(iterator8, 2))
    assert_equal(occ8[4],
                 (DiatomicSpinOrbital(n=1, l=3, ml=3, spin='beta'),
                  DiatomicSpinOrbital(n=1, l=3, ml=-3, spin='beta')))
    assert_equal(len(occ8), 6)


def test_calc_vals_and_term_symbol():
    iterator = atomic_spinorbitals_iterator(1, 0)
    occ1 = list(occupy(iterator, 1))
    assert_equal(calc_vals(occ1[0]), (0, Frac(1, 2)))
    assert_equal(calc_vals(occ1[1]), (0, Frac(-1, 2)))
    assert_equal(find_term_symbol(occ1[0]), TermSymbol(2, 0))

    orbs1 = [AtomicSpinOrbital(n=3, l=2, ml=-2, spin='alpha'),
             AtomicSpinOrbital(n=5, l=4, ml=-1, spin='beta')]
    assert_equal(calc_vals(orbs1), (-3, 0))
    assert_equal(find_term_symbol(orbs1), TermSymbol(1, 3))


def test_atomic_terms_table():
    a = AtomicTermTable(2, 1)
    a.set(1, 1, 5)
    assert_equal(a.get(1, 1), 5)

    s2_table = AtomicTermTable(1, 0)
    s2_table.set(1, 0, 1)
    s2_terms = subshell_terms('atomic', 1, 0, 2)
    assert_equal(s2_terms, s2_table)
    np_assert_equal(s2_terms.cleaned().table, [[1]])

    p2_terms = subshell_terms('atomic', 2, 1, 2)
    p2_table = np.array([[3, 2, 1], [1, 1, 0]])
    np_assert_equal(p2_terms.table, p2_table)
    np_assert_equal(p2_terms.cleaned().table, [[1, 0, 1], [0, 1, 0]])


def test_diatomic_term_table():
    d = DiatomicTermTable(2, 1)
    d.set(1, 1, 5)
    assert_equal(d.get(1, 1), 5)
    pi2_terms = subshell_terms('diatomic', 1, 1, 2)
    np_assert_equal(pi2_terms.cleaned().table, [[1, 0, 1], [1, 0, 0]])


def test_terms_table_string():
    s2_string = '\\begin{tabular}{ r | c } \nM\\L &          0 \\hl \n  1 &  ' \
                + '\\O{$^1$S} \\\\ \n\\end{tabular}'
    s2_terms = subshell_terms('atomic', 3, 0, 2)
    assert_equal(s2_terms.cleaned().string('latex-crossed'), s2_string)


# def test_mul():
#    two_s1_terms = subshell_terms(2, 0, 1)
#    three_s1_terms = subshell_terms(3, 0, 1)
#    print(two_s1_terms.string())
#    print(three_s1_terms.string())
#    print(two_s1_terms * three_s1_terms)
#    print(multiple_subshell_terms((2, 0, 1), (3, 0, 1)).cleaned())
#
#    two_p1_terms = subshell_terms(2, 1, 1).cleaned()
#    three_p1_terms = subshell_terms(3, 1, 1).cleaned()
#    print(two_p1_terms.string())
#    print(three_p1_terms.string())
#    mul = two_p1_terms * three_p1_terms
#    print(mul)
#    multi_subshell = multiple_subshell_terms((2, 1, 1), (3, 1, 1))
#    print(multi_subshell)
#    np_assert_equal(mul.table, multi_subshell.cleaned().table)


def test_multiple_atomic_subshell_terms():
    mult_1s2 = multiple_subshell_terms('atomic', (1, 0, 2))
    assert_equal(mult_1s2.max_mult, 1)
    assert_equal(mult_1s2.max_am, 0)
    np_assert_equal(mult_1s2.table, [[1]])
    np_assert_equal(mult_1s2.table, [[1]])

    mult_1s1_2s1_3s2 = multiple_subshell_terms('atomic', (1, 0, 1), (2, 0, 1),
                                               (3, 0, 2))
    assert_equal(mult_1s1_2s1_3s2.max_mult, 3)
    assert_equal(mult_1s1_2s1_3s2.max_am, 0)
    table = [[2], [1]]
    np_assert_equal(mult_1s1_2s1_3s2.table, table)
    cleaned = [[1], [1]]
    np_assert_equal(mult_1s1_2s1_3s2.cleaned().table, cleaned)


def test_multiple_diatomic_subshell_terms():
    mult_1s2 = multiple_subshell_terms('diatomic', (1, 0, 2))
    assert_equal(mult_1s2.max_mult, 1)
    assert_equal(mult_1s2.max_am, 0)
    np_assert_equal(mult_1s2.table, [[1]])
    np_assert_equal(mult_1s2.table, [[1]])

    mult_1s1_1p2_3d3 = multiple_subshell_terms('diatomic', (1, 0, 1), (2, 1, 2),
                                               (3, 2, 3))
    assert_equal(mult_1s1_1p2_3d3.max_mult, 5)
    assert_equal(mult_1s1_1p2_3d3.max_am, 4)
    table = [[4, 0, 6, 0, 2], [2, 0, 4, 0, 1], [0, 0, 1, 0, 0]]
    np_assert_equal(mult_1s1_1p2_3d3.table, table)
    cleaned = [[2, 0, 2, 0, 1], [2, 0, 3, 0, 1], [0, 0, 1, 0, 0]]
    np_assert_equal(mult_1s1_1p2_3d3.cleaned().table, cleaned)

#def test_mult_long():
#    mult_1s1_2p2_3d3 = multiple_subshell_terms((1, 0, 1), (2, 1, 2), (3, 2, 3))
#    assert_equal(mult_1s1_2p2_3d3.max_mult, 7)
#    assert_equal(mult_1s1_2p2_3d3.max_am, 8)
#    t = np.array([[196, 182, 148, 102,  60,  28,  10,   2,   0],
#                  [138, 128, 103,  70,  40,  18,   6,   1,   0],
#                  [46,   42,  33,  21,  11,   4,   1,   0,   0],
#                  [6,     5,   4,   2,   1,   0,   0,   0,   0]])
#    np_assert_equal(mult_1s1_2p2_3d3.table, t)
#
#    c = [[4,  9, 13, 12, 10,  6,  3,  1,  0],
#         [6, 16, 21, 20, 15,  9,  4,  1,  0],
#         [3,  8, 10,  9,  6,  3,  1,  0,  0],
#         [1,  1,  2,  1,  1,  0,  0,  0,  0]]
#    np_assert_equal(mult_1s1_2p2_3d3.cleaned().table, c)


def test_all_atomic_term_tables():
    tables = all_atomic_term_tables(2)
    one_s1 = subshell_terms('atomic', 1, 0, 1)
    assert_equal(next(tables), one_s1)
    two_p1 = subshell_terms('atomic', 2, 1, 1)
    assert_equal(next(tables), two_p1)
    two_p2 = subshell_terms('atomic', 2, 1, 2)
    assert_equal(next(tables), two_p2)
