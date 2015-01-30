from nose.tools import *
from quantum.term_symbols import *
from fractions import Fraction
from numpy.testing import assert_equal as np_assert_equal


def setup():
    pass


def teardown():
    pass


def test_spinorbital():
    one_s1a = SpinOrbital(1, 0, 0, 1)
    assert_equal(one_s1a.__repr__(), '1s_{0}a')
    five_g2b = SpinOrbital(5, 4, 2, 'beta')
    assert_equal(five_g2b.__repr__(), '5g_{2}b')
    a = one_s1a
    assert_raises(SyntaxError, SpinOrbital.__init__, a, 2, 2, 1, 1)
    assert_raises(SyntaxError, SpinOrbital.__init__, a, 2, 1, 2, 1)
    assert_raises(SyntaxError, SpinOrbital.__init__, a, 2, 1, 1, 'q')


def test_spin_iterator():
    a = Fraction(1, 2)
    b = Fraction(-1, 2)
    assert_equal(list(spin_iterator()), [a, b])


def test_spin_orbitals_iterator():
    p_orbs = ['2p_{1}a', '2p_{1}b', '2p_{0}a', '2p_{0}b', '2p_{-1}a', '2p_{-1}b']
    p_l = list(map(str, spin_orbitals_iterator(2, 1)))
    assert_equal(p_l, p_orbs)


def test_occupy():
    assert_equal(list(occupy(1, 0, 0)), [()])
    assert_equal(list(occupy(2, 1, 0)), [()])
    assert_equal(list(occupy(2, 1, 0)), [()])

    occ1 = list(occupy(1, 0, 1))
    assert_equal(occ1[0][0], SpinOrbital(1, 0, 0, 'alpha'))
    assert_equal(len(occ1), 2)

    occ2 = list(occupy(1, 0, 2))
    assert_equal(occ2[0][0], SpinOrbital(1, 0, 0, 'alpha'))
    assert_equal(len(occ2), 1)

    occ3 = list(occupy(2, 1, 2))
    assert_equal(len(occ3), 15)


def test_calc_vals_and_term_symbol():
    occ1 = list(occupy(1, 0, 1))
    assert_equal(calc_vals(occ1[0]), (0, Frac(1, 2)))
    assert_equal(calc_vals(occ1[1]), (0, Frac(-1, 2)))
    assert_equal(find_term_symbol(occ1[0]), TermSymbol(2, 0))

    orbs1 = [SpinOrbital(3, 2, -2, 'alpha'), SpinOrbital(5, 4, -1, 'beta')]
    assert_equal(calc_vals(orbs1), (-3, 0))
    assert_equal(find_term_symbol(orbs1), TermSymbol(1, 3))


def test_terms_table():
    a = TermTable(2, 1)
    a.set(1, 1, 5)
    assert_equal(a.get(1, 1), 5)


def test_subshell_terms_and_clean():
    s2_table = TermTable(1, 0)
    s2_table.set(1, 0, 1)
    s2_terms = subshell_terms(1, 0, 2)
    assert_equal(s2_terms, s2_table)
    s2_terms.clean_table()
    assert_equal(s2_terms, s2_table)

    p2_table = TermTable(2, 1)
    p2_table.table = np.array([[3, 2, 1], [1, 1, 0]])
    p2_terms = subshell_terms(2, 1, 2)
    np_assert_equal(p2_terms.table, p2_table.table)
    p2_terms.clean_table()
    p2_table.table = np.array([[1, 0, 1], [0, 1, 0]])
    np_assert_equal(p2_terms.table, p2_table.table)