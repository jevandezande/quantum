from quantum.point_group import *
import nose
from nose.tools import *
import numpy as np
from numpy import pi
from numpy.testing import assert_allclose
from numpy.linalg import norm


def test_repr_str():
    assert_equal(repr(C2v), '<PointGroup C2v>')
    c2v_str = '''\
---------------------------------------------------------------------
| C2v |  E  | C2  |σ(xz)|σ(yz)|    Lin Rot     |        Quad        |
---------------------------------------------------------------------
| A1  |  1  |  1  |  1  |  1  |       z        |     x2, y2, z2     |
| A2  |  1  |  1  | -1  | -1  |       Rz       |         xy         |
| B1  |  1  | -1  | -1  |  1  |     x, Ry      |         xz         |
| B2  |  1  | -1  |  1  | -1  |     y, Rx      |         yz         |
---------------------------------------------------------------------
'''
    assert_equal(str(C2v), c2v_str)


def test_order():
    assert_equal(C1.order, 1)
    assert_equal(C2.order, 2)
    assert_equal(Oh.order, 48)


def test_latex():
    D3h_latex = r'''\begin{tabular}{l c c c c c c c c}\hl
D3h  &  E  & C3  & C'2 &{\sigma}h& S3  &{\sigma}v&    Lin Rot     &        Quad        \hl
A'1  &  1  &  1  &  1  &  1  &  1  &  1  &            &   x2+y2, z2     \\
A'2  &  1  &  1  & -1  &  1  &  1  & -1  &     Rz     &                 \\
E'   &  2  & -1  &  0  &  2  & -1  &  0  &   (x, y)   &  (x2-y2, xy)    \\
A"1  &  1  &  1  &  1  & -1  & -1  & -1  &            &                 \\
A"2  &  1  &  1  & -1  & -1  & -1  &  1  &     z      &                 \\
E"   &  2  & -1  &  0  & -2  &  1  &  0  &  (Rx, Ry)  &    (xz, yz)     \\
\hl
\end{tabular}'''
    assert_equal(D3h.latex(), D3h_latex)


def test_orthogonality():
    for pg in pg_list:
        # Cn are formulated differently, and thus orthogonality code does not work
        if pg.name in ['C3', 'C4', 'C5']:
            continue
        if not PointGroup.check_orthogonality(pg):
            print(pg)
            raise Exception(f'{pg.name} is not orthogonal')


def test_reduce():
    gamma = np.array([4])
    reduction = reduce(gamma, C1)
    assert_equal(reduction, [(4, 'A')])

    gamma = np.array([12, 0, 0])
    reduction = reduce(gamma, C3)
    assert_equal(reduction, [(4, 'A'), (4, 'E_a'), (4, 'E_b')])

    gamma = np.array([15, -1, 3, 3])
    reduction = reduce(gamma, C2v)
    assert_equal(reduction, [(5, 'A1'), (2, 'A2'), (4, 'B1'), (4, 'B2')])

    gamma = np.array([10, -3, 1, 2])
    assert_raises(ValueError, reduce, gamma, C2v)


def test_vibrations():
    gamma = np.array([7])
    vibs = vibrations(gamma, C1)
    assert_equal(vibs, [(1, 'A')])

    gamma = np.array([12, 0, 0])
    vibs = vibrations(gamma, C3)
    assert_equal(vibs, [(2, 'A'), (2, 'E_a'), (2, 'E_b')])
    assert_equal(total_vibrations(vibs), 6)
    ir, raman = classify_vibrations(vibs, C3)

    gamma = np.array([15, -1, 3, 3])
    vibs = vibrations(gamma, C2v)
    assert_equal(vibs, [(4, 'A1'), (1, 'A2'), (2, 'B1'), (2, 'B2')])
    assert_equal(total_vibrations(vibs), 9)
    ir, raman = classify_vibrations(vibs, C2v)
    assert_equal(ir, [(4, 'A1'), (2, 'B1'), (2, 'B2')])
    assert_equal(raman, [(4, 'A1'), (1, 'A2'), (2, 'B1'), (2, 'B2')])

    gamma = [54,  0,  0,  0,  0,  2]
    vibs = vibrations(gamma, D3d)
    assert_equal(vibs, [(5, 'A1g'), (3, 'A2g'), (8, 'Eg'), (4, 'A1u'), (4, 'A2u'), (8, 'Eu')])
    assert_equal(total_vibrations(vibs), 48)
    ir, raman = classify_vibrations(vibs, D3d)
    assert_equal(ir, [(4, 'A2u'), (8, 'Eu')])
    assert_equal(raman, [(5, 'A1g'), (8, 'Eg')])

    gamma = [12,  0, -2,  4, -2,  2]
    vibs = vibrations(gamma, D3h)
    assert_equal(vibs, [(1, "A'1"), (0, "A'2"), (2, "E'"), (0, 'A"1'), (1, 'A"2'), (0, 'E"')])
    assert_equal(total_vibrations(vibs), 6)
    ir, raman = classify_vibrations(vibs, D3h)
    assert_equal(ir, [(2, "E'"), (1, 'A"2')])
    assert_equal(raman, [(1, "A'1"), (2, "E'")])


def test_degeneracy():
    assert_equal(PointGroup.degeneracy('B'), 1)
    assert_equal(PointGroup.degeneracy('E'), 2)
    assert_equal(PointGroup.degeneracy('H'), 5)


def test_pg_same_irrep():
    pg_same_irrep = PointGroup.pg_same_irrep
    assert_true(pg_same_irrep(C1, 0, 0))  # A, A
    assert_false(pg_same_irrep(C2v, 2, 3))  # B1, B2
    assert_true(pg_same_irrep(C3, 1, 2))  # E_a, E_b
    assert_false(pg_same_irrep(Oh, 4, 9))  # T2g, T2u


if __name__ == '__main__':
    import sys

    module_name = sys.modules[__name__].__file__
    nose.run(argv=[sys.argv[0], module_name, '-v'])
