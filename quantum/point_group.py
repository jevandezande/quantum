import numpy as np
from numpy import pi as π, cos, exp


class PointGroup:
    def __init__(self, name, ops, coeffs, irreps, table, lin_rot, quad):
        """
        :param name: the name of the point group
        :param ops: list of symmetry operation names
        :param coeffs: coefficients of the irreps
        :param irreps: list of irreducible representation names
        :param table: characters of the symmetry operations in the irreducible representation
        :param lin_rot: linear and rotation symmetry transformations
        :param quad: quadratic symmetry transformations
        """
        if not (len(irreps), len(ops)) == table.shape:
            raise Exception(f'Invalid size, table shape does not match irreps x ops: ' +
                            f'{table.shape} != {len(irreps)} x {len(ops)}')
        if not len(coeffs) == len(ops):
            raise Exception('Mismatched lengths of coefficients and operations.')

        self.name = name
        self.ops = ops
        self.coeffs = coeffs
        self.order = sum(coeffs)
        self.irreps = irreps
        self.table = table
        self.lin_rot = []
        for lr in lin_rot:
            if not lr:
                lr = tuple()
            elif not isinstance(lr, tuple):
                lr = (lr, )
            self.lin_rot.append(lr)
        self.quad = []
        for q in quad:
            if not isinstance(q, tuple):
                if len(q) == 0:
                    q = tuple()
                else:
                    q = (q, )
            self.quad.append(q)

    def __repr__(self):
        return f'<PointGroup {self.name}>'

    def __str__(self):
        """
        Generate a nice string of the table
        """
        line_form = '|{:5s}|' + '{:^5g}|'*len(self.ops)
        out = f'|{self.name:5s}|' + ('{:^5s}|'*len(self.ops)).format(*self.ops) + '    Lin Rot     |        Quad        |\n'
        horiz_line = '-'*(len(out) - 1) + '\n'
        out = horiz_line + out + horiz_line
        for irrep, line, lin_rot, quad in zip(self.irreps, self.table, self.lin_rot, self.quad):
            out += line_form.format(irrep, *line)
            lin_rot = str(lin_rot)[1:-1].strip(',').replace("'", '')
            quad = str(quad)[1:-1].strip(',').replace("'", '')
            out += f'{lin_rot:^16s}|{quad:^20s}|\n'
        return out + horiz_line

    def __iter__(self):
        """
        Iterate through the irreps
        """
        for irrep, line, lin_rot, quad in zip(self.irreps, self.table, self.lin_rot, self.quad):
            yield irrep, line, lin_rot, quad

    def latex(self):
        """
        Generate a string of the table in latex form
        """
        # clean ops
        ops = []
        for op in self.ops:
            ops.append(op.replace('σ', '\\sigma '))

        line_form = '{:5s}&' + '{:^5d}&'*len(ops)
        out = '\\begin{tabular}{l' + ' c'*(len(ops) + 2) + '}\\hl\n'
        out += f'{self.name:5s}&' + ('{:^5s}&'*len(ops)).format(*ops) + '    Lin Rot     &        Quad        \\hl\n'
        for irrep, line, lin_rot, quad in zip(self.irreps, self.table, self.lin_rot, self.quad):
            out += line_form.format(irrep, *line)
            lin_rot = str(lin_rot)[1:-1].strip(',').replace("'", '')
            quad = str(quad)[1:-1].strip(',').replace("'", '')
            out += f'{lin_rot:^12s}&{quad:^16s} \\\\\n'
        return out.rstrip('\\') + '\\hl\n\\end{tabular}'

    def irrep(self, name):
        """
        Find the specified irrep and return its values
        """
        try:
            idx = self.irreps.index(name)
        except ValueError as e:
            raise Exception('Invalid irrep.')
        return self.table[idx], self.lin_rot[idx], self.quad[idx]

    @staticmethod
    def degeneracy(irrep):
        letter = irrep[0]
        if letter in ['A', 'B']:
            return 1
        else:
            # degen:  2345
            return '  ETGH'.index(letter)

    @staticmethod
    def pg_same_irrep(pg, i, j):
        """
        Checks if the indices refer to the same irrep, even if the sub-irrep
        is different.

        :param pg: a PointGroup object
        :param i, j: indices into pg
        :return: True or False
        """
        if i == j:
            return True
        a = pg.irreps[i]
        b = pg.irreps[j]
        # If it has a sub-irrep (e.g. Eg_a)
        if len(a) > 2 and a[-2] == '_' and len(b) > 2 and b[-2] == '_':
            # if same irrep
            if a[:-1] == b[:-1]:
                return True
        return False

    @staticmethod
    def check_orthogonality(pg):
        """
        Check if the given point group in orthogonal
        """
        for i, line1 in enumerate(pg.table):
            for j, line2 in enumerate(pg.table):
                val = 1/pg.order * sum(pg.coeffs * line1 * line2)
                target = 0
                if PointGroup.pg_same_irrep(pg, i, j):
                    target = 1
                if not np.isclose(val, target):
                    return False
        return True


def reduce(gamma, pg):
    """
    Use the reduction formula to convert a gamma into irreps
    Note: reduce is a python2 built-in, but was removed in python3
    """
    reduction = []
    for irrep, line, *_ in pg:
        val = 1/pg.order * sum(gamma * line * pg.coeffs)
        real = val.real
        imag = val.imag
        if abs(real - round(real)) > 1e-10 or imag > 1e-10:
            raise Exception('Failed reduction, non-integer returned: {val} Irrep: {irrep}')
        reduction.append((int(round(real)), irrep))
    return reduction


def vibrations(gamma, pg):
    """
    Determine the vibrations from a gamma
    """
    reduction = reduce(gamma, pg)
    vibs = []
    for val, irrep in reduction:
        line, lin_rot, quad = pg.irrep(irrep)
        val -= len(lin_rot)
        vibs.append((val, irrep))
    return vibs


def total_vibrations(vibs):
    total = 0
    for val, irrep in vibs:
        # TODO: Fix hack for C3
        if len(irrep) > 2 and irrep[-2] == '_':
            total += val
        else:
            total += val*PointGroup.degeneracy(irrep)
    return total


def ir_active(irrep, pg):
    """
    Return True if IR active irrep
    """
    lines, lin_rot, quad = pg.irrep(irrep)

    if len(lin_rot) == 0:
        return False

    if 'x' in lin_rot or 'y' in lin_rot or 'z' in lin_rot:
        return True

    for lr in lin_rot:
        if isinstance(lr, tuple):
            if 'x' in lr or 'y' in lr or 'z' in lr:
                return True
    return False


def raman_active(irrep, pg):
    """
    Return True if Raman active irrep
    """
    lines, lin_rot, quad = pg.irrep(irrep)
    return len(quad) > 0


def classify_vibrations(vibs, pg):
    """
    Classify the vibrations as IR and/or Raman active

    :param vibs: [(val, irrep), ...]
    :param pg: PointGroup
    :return: ir active [(val, irrep), ...], raman active [(val, irrep), ...]
    """
    ir = []
    raman = []
    for val, irrep in vibs:
        if val == 0:
            continue
        if ir_active(irrep, pg):
            ir.append((val, irrep))
        if raman_active(irrep, pg):
            raman.append((val, irrep))
    return ir, raman


################
# Point Groups #
################

#############
# Non-axial #
#############

c1_table = np.array([[1]])
c1_lin_rot = [('x', 'y', 'z', 'Rx', 'Ry', 'Rz')]
c1_quad = [('x2', 'y2', 'z2', 'xy', 'xz', 'yz')]
C1 = PointGroup('C1', ['E'], np.array([1]), ['A'], c1_table, c1_lin_rot, c1_quad)


cs_table = np.array([[1, 1],[1, -1]])
cs_ops = ['E', 'σ_h']
cs_coeffs = [1, 1]
cs_irreps = ["A'", 'A"']
cs_lin_rot = [('x', 'y', 'Rz'), ('z', 'Rx', 'Ry')]
cs_quad = [('x2', 'y2', 'z2', 'xy'), ('xz', 'yz')]
Cs = PointGroup('Cs', cs_ops, cs_coeffs, cs_irreps, cs_table, cs_lin_rot, cs_quad)


ci_table = np.array([[1, 1],[1, -1]])
ci_ops = ['E', 'i']
ci_coeffs = [1, 1]
ci_irreps = ['Ag', 'Au']
ci_lin_rot = [('Rx', 'Ry', 'Rz'), ('x', 'y', 'z')]
ci_quad = [('x2', 'y2', 'z2', 'xy', 'xz', 'yz'), ('',)]
Ci = PointGroup('Ci', ci_ops, ci_coeffs, ci_irreps, ci_table, ci_lin_rot, ci_quad)

######
# Cn #
######
c2_table = np.array([[1, 1],[1, -1]])
c2_ops = ['E', 'C2']
c2_coeffs = [1, 1]
c2_irreps = ['A', 'B']
c2_lin_rot = [('z', 'Rz'), ('x', 'y', 'Rx', 'Ry')]
c2_quad = [('x2', 'y2', 'z2', 'xy'), ('xz', 'yz')]
C2 = PointGroup('C2', c2_ops, c2_coeffs, c2_irreps, c2_table, c2_lin_rot, c2_quad)


# 2j == 2*sqrt(-1)
e = exp(2j*π/3)
es = np.conj(e)
c3_table = np.array([
    [1,  1,  1],
    [1,  e, es],
    [1, es,  e]
])
c3_ops = ['E', 'C3', 'C3^2']
c3_coeffs = [1, 1, 1]
c3_irreps = ['A', 'E_a', 'E_b']
c3_lin_rot = [('z', 'Rz'), ('x+iy', 'Rx+iRy'), ('x-iy', 'Rx-iRy')]
c3_quad = [('x2+y2', 'z2'), (('x2-y2', 'xy'),), (('xz', 'yz'),)]
C3 = PointGroup('C3', c3_ops, c3_coeffs, c3_irreps, c3_table, c3_lin_rot, c3_quad)


c4_table = np.array([
    [1,  1,  1,  1],
    [1, -1,  1, -1],
    [1, 1j, -1,-1j],
    [1,-1j, -1, 1j]
])
c4_ops = ['E', 'C4', 'C2', 'C4^3']
c4_coeffs = [1, 1, 1, 1]
c4_irreps = ['A', 'B', 'E_a', 'E_b']
c4_lin_rot = [('z', 'Rz'), '', ('x+iy', 'Rx+iRy'), ('x-iy', 'Rx-iRy')]
c4_quad = [('x2+y2', 'z2'), ('x2-y2', 'xy'), 'xz', 'yz']
C4 = PointGroup('C4', c4_ops, c4_coeffs, c4_irreps, c4_table, c4_lin_rot, c4_quad)

e = exp(2j*π/5)
es = np.conj(e)
c5_table = np.array([
    [1,  1,     1,      1,      1   ],
    [1,  e,    e**2,   es**2,  es   ],
    [1, es,    es**2,  e**2,    e   ],
    [1, e**2,   es,     e,     es**2],
    [1, es**2,  e,      es,    e**2 ],
])
c5_ops = ['E', 'C5', 'C5^2', 'C5^3', 'C5^4']
c5_coeffs = [1, 1, 1, 1, 1]
c5_irreps = ['A', 'E_1a', 'E_1b', 'E_2a', 'E_2b']
c5_lin_rot = [('z', 'Rz'), ('x+iy', 'Rx+iRy'), ('x-iy', 'Rx-iRy'), (), ()]
c5_quad = [('x2+y2', 'z2'), (('xz', 'yz'),), (),  (('x2-y2', 'xy'),), ()]
C5 = PointGroup('C5', c5_ops, c5_coeffs, c5_irreps, c5_table, c5_lin_rot, c5_quad)


#######
# Cnv #
#######
c2v_table = np.array([
    [ 1,  1,  1,  1],
    [ 1,  1, -1, -1],
    [ 1, -1, -1,  1],
    [ 1, -1,  1, -1]
])
c2v_ops = ['E', 'C2', 'σ(xz)', 'σ(yz)']
c2v_coeffs = [1, 1, 1, 1]
c2v_irreps = ['A1', 'A2', 'B1', 'B2']
c2v_lin_rot = ['z', 'Rz', ('x', 'Ry'), ('y', 'Rx')]
c2v_quad = [('x2', 'y2', 'z2'), 'xy', 'xz', 'yz']
C2v = PointGroup('C2v', c2v_ops, c2v_coeffs, c2v_irreps, c2v_table, c2v_lin_rot, c2v_quad)


c3v_table = np.array([
    [ 1,  1,  1],
    [ 1,  1, -1],
    [ 2, -1,  0]
])
c3v_ops = ['E', 'Cz', 'σv']
c3v_coeffs = [1, 2, 3]
c3v_irreps = ['A1', 'A2', 'E']
c3v_lin_rot = ['z', 'Rz', (('x', 'y'), ('Rx', 'Ry'))]
c3v_quad = [('x2+y2', 'z2'), '', (('x2-y2', 'xy'), ('xz', 'yz'))]
C3v = PointGroup('C3v', c3v_ops, c3v_coeffs, c3v_irreps, c3v_table, c3v_lin_rot, c3v_quad)


#######
# Cnh #
#######





######
# Dn #
######





#######
# Dnh #
#######
d2h_table = np.array([
    [ 1,  1,  1,  1,  1,  1,  1,  1],
    [ 1,  1, -1, -1,  1,  1, -1, -1],
    [ 1, -1,  1, -1,  1, -1,  1, -1],
    [ 1, -1, -1,  1,  1, -1, -1,  1],
    [ 1,  1,  1,  1, -1, -1, -1, -1],
    [ 1,  1, -1, -1, -1, -1,  1,  1],
    [ 1, -1,  1, -1, -1,  1, -1,  1],
    [ 1, -1, -1,  1, -1,  1,  1, -1],
])
d2h_ops = ['E', 'C2(z)', 'C2(y)', 'C2(x)', 'i', 'σ(xy)', 'σ(xz)', 'σ(yz)']
d2h_coeffs = [1, 1, 1, 1, 1, 1, 1, 1]
d2h_irreps = ['Ag', 'B1g', 'B2g', 'B3g', 'Au', 'B1u', 'B2u', 'B3u']
d2h_lin_rot = ['', 'Rz', 'Ry', 'Rx', '', 'z', 'y', 'x']
d2h_quad = [('x2', 'y2', 'z2'), 'xy', 'xz', 'yz', '', '', '', '']
D2h = PointGroup('D2h', d2h_ops, d2h_coeffs, d2h_irreps, d2h_table, d2h_lin_rot, d2h_quad)

d3h_table = np.array([
    [ 1,  1,  1,  1,  1,  1],
    [ 1,  1, -1,  1,  1, -1],
    [ 2, -1,  0,  2, -1,  0],
    [ 1,  1,  1, -1, -1, -1],
    [ 1,  1, -1, -1, -1,  1],
    [ 2, -1,  0, -2,  1,  0]
])
d3h_ops = ['E', 'C3', "C'2", 'σh', 'S3', 'σv']
d3h_coeffs = [1, 2, 3, 1, 2, 3]
d3h_irreps = ["A'1", "A'2", "E'", 'A"1', 'A"2', 'E"']
d3h_lin_rot = ['', 'Rz', (('x', 'y'),), '', 'z', (('Rx', 'Ry'),)]
d3h_quad = [('x2+y2', 'z2'), '', (('x2-y2', 'xy'),), '', '', (('xz', 'yz'),)]
D3h = PointGroup('D3h', d3h_ops, d3h_coeffs, d3h_irreps, d3h_table, d3h_lin_rot, d3h_quad)


d4h_table = np.array([
    [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    [ 1,  1,  1, -1, -1,  1,  1,  1, -1, -1],
    [ 1, -1,  1,  1, -1,  1, -1,  1,  1, -1],
    [ 1, -1,  1, -1,  1,  1, -1,  1, -1,  1],
    [ 2,  0, -2,  0,  0,  2,  0, -2,  0,  0],
    [ 1,  1,  1,  1,  1, -1, -1, -1, -1, -1],
    [ 1,  1,  1, -1, -1, -1, -1, -1,  1,  1],
    [ 1, -1,  1,  1, -1, -1,  1, -1, -1,  1],
    [ 1, -1,  1, -1,  1, -1,  1, -1,  1, -1],
    [ 2,  0, -2,  0,  0, -2,  0,  2,  0,  0]
])
d4h_ops = ['E', 'C4', 'C2', "C'2", 'C"2', 'i', 'S4', 'σh', 'σv', 'σd']
d4h_coeffs = [1, 2, 1, 2, 2, 1, 2, 1, 2, 2]
d4h_irreps = ['A1g', 'A2g', 'B1g', 'B2g', 'Eg', 'A1u', 'A2u', 'B1u', 'B2u', 'Eu']
d4h_lin_rot = ['', 'Rz', '', '', (('Rx', 'Ry'),), '', 'z', '', '', (('x', 'y'),)]
d4h_quad = [('x2+y2', 'z2'), '', 'x2-y2', 'xy', (('xz', 'yz'),), '', '', '', '', '']
D4h = PointGroup('D4h', d4h_ops, d4h_coeffs, d4h_irreps, d4h_table, d4h_lin_rot, d4h_quad)


#######
# Dnd #
#######

d2d_table = np.array([
    [ 1,  1,  1,  1,  1],
    [ 1,  1,  1, -1, -1],
    [ 1, -1,  1,  1, -1],
    [ 1, -1,  1, -1,  1],
    [ 2,  0, -2,  0,  0]
])
d2d_ops = ['E', 'S4', 'C2', "C'2", 'σd']
d2d_coeffs = [1, 2, 1, 2, 2]
d2d_irreps = ['A1', 'A2', 'B1', 'B2', 'E']
d2d_lin_rot = ['', 'Rz', '', 'z', (('x', 'y'), ('Rx', 'Ry'),)]
d2d_quad = [('x2+y2', 'z2'), '', 'x2-y2', 'xy', (('xz', 'yz'),)]
D2d = PointGroup('D2d', d2d_ops, d2d_coeffs, d2d_irreps, d2d_table, d2d_lin_rot, d2d_quad)


d3d_table = np.array([
    [ 1,  1,  1,  1,  1,  1],
    [ 1,  1, -1,  1,  1, -1],
    [ 2, -1,  0,  2, -1,  0],
    [ 1,  1,  1, -1, -1, -1],
    [ 1,  1, -1, -1, -1,  1],
    [ 2, -1,  0, -2,  1,  0]
])
d3d_ops = ['E', 'C3', "C'2", 'i', 'S6', 'σd']
d3d_coeffs = [1, 2, 3, 1, 2, 3]
d3d_irreps = ['A1g', 'A2g', 'Eg', 'A1u', 'A2u', 'Eu']
d3d_lin_rot = ['', 'Rz', (('Rx', 'Ry'),), '', 'z', (('x', 'y'),)]
d3d_quad = [('x2+y2', 'z2'), '', (('x2-y2', 'xy'), ('xz', 'yz')), '', '', '']
D3d = PointGroup('D3d', d3d_ops, d3d_coeffs, d3d_irreps, d3d_table, d3d_lin_rot, d3d_quad)




######
# Sn #
######





##########
# Higher #
##########

td_table = np.array([
    [ 1,  1,  1,  1,  1],
    [ 1,  1,  1, -1, -1],
    [ 2, -1,  2,  0,  0],
    [ 3,  0, -1,  1, -1],
    [ 3,  0, -1, -1,  1]
])
td_ops = ['E', 'C3', 'C2', 'S4', 'σd']
td_coeffs = [1, 8, 3,  6, 6]
td_irreps = ['A1', 'A2', 'E', 'T1', 'T2']
td_lin_rot = ['', '', '', (('Rx', 'Ry', 'Rz'),), (('x', 'y', 'z'),)]
td_quad = ['x2+y2+z2', '', (('2z2-x-2y2', 'x2-y2'),), '', (('xy', 'xz', 'yz'),)]
Td = PointGroup('Td', td_ops, td_coeffs, td_irreps, td_table, td_lin_rot, td_quad)


oh_table = np.array([
    [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    [ 1,  1, -1, -1,  1,  1, -1,  1,  1, -1],
    [ 2, -1,  0,  0,  2,  2,  0, -1,  2,  0],
    [ 3,  0, -1,  1, -1,  3,  1,  0, -1, -1],
    [ 3,  0,  1, -1, -1,  3, -1,  0, -1,  1],
    [ 1,  1,  1,  1,  1, -1, -1, -1, -1, -1],
    [ 1,  1, -1, -1,  1, -1,  1, -1, -1,  1],
    [ 2, -1,  0,  0,  2, -2,  0,  1, -2,  0],
    [ 3,  0, -1,  1, -1, -3, -1,  0,  1,  1],
    [ 3,  0,  1, -1, -1, -3,  1,  0,  1, -1]
])
oh_ops = ['E', 'C3', 'C2', 'C4', "C4^2", 'i', 'S4', 'S6', 'σh', 'σd']
oh_coeffs = [1, 8, 6, 6, 3, 1, 6, 8, 3, 6]
oh_irreps = ['A1g', 'A2g', 'Eg', 'T1g', 'T2g', 'A1u', 'A2u', 'Eu', 'T1u', 'T2u']
oh_lin_rot = ['', '', '', (('Rx', 'Ry', 'Rz'),), '', '', '', '', (('x', 'y', 'z'),), '']
oh_quad = ['x2+y2+z2', '', (('2z2-x-2y2', 'x2-y2'),), '', (('xy', 'xz', 'yz'),), '', '', '', '', '']
Oh = PointGroup('Oh', oh_ops, oh_coeffs, oh_irreps, oh_table, oh_lin_rot, oh_quad)


c2 = 2*cos(2*π/5)
c4 = 2*cos(4*π/5)
ih_table = np.array([
    [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    [ 3,-c4,-c2,  0, -1,  3,-c2,-c4,  0, -1],
    [ 3,-c2,-c4,  0, -1,  3,-c4,-c2,  0, -1],
    [ 4, -1, -1,  1,  0,  4, -1, -1,  1,  0],
    [ 5,  0,  0, -1,  1,  5,  0,  0, -1,  1],
    [ 1,  1,  1,  1,  1, -1, -1, -1, -1, -1],
    [ 3,-c4,-c2,  0, -1, -3, c2, c4,  0,  1],
    [ 3,-c2,-c4,  0, -1, -3, c4, c2,  0,  1],
    [ 4, -1, -1,  1,  0, -4,  1,  1, -1,  0],
    [ 5,  0,  0, -1,  1, -5,  0,  0,  1, -1]
])
ih_ops = ['E', 'C5', 'C5^2', 'C3', 'C2', 'i', 'S4', 'S6', 'σh', 'σd']
ih_coeffs = [1, 12, 12, 20, 15, 1, 12, 12, 20, 15]
ih_irreps = ['Ag', 'T1g', 'T2g', 'Gg', 'Hg', 'Au', 'T1u', 'T2u', 'Gu', 'Hu']
ih_lin_rot = ['', (('Rx', 'Ry', 'Rz'),), '', '', '', '', (('x', 'y', 'z'),), '', '', '', '']
ih_quad = ['x2+y2+z2', '', '', '', (('2z2-x-2y2', 'x2-y2', 'xy', 'xz', 'yz'),), '', '', '', '', '']
Ih = PointGroup('Ih', ih_ops, ih_coeffs, ih_irreps, ih_table, ih_lin_rot, ih_quad)


pg_list = [C1, Ci, Cs, C2, C3, C4, C2v, C3v, D2h, D3h, D4h, D2d, D3d, Td, Oh, Ih]
pg_dict = {
    'C1': C1,
    'Ci': Ci,
    'Cs': Cs,
    'C2': C2,
    'C3': C3,
    'C4': C4,
    'C2v': C2v,
    'C3v': C3v,
    'D2h': D2h,
    'D3h': D3h,
    'D4h': D4h,
    'D2d': D2d,
    'D3d': D3d,
    'Td': Td,
    'Oh': Oh,
    'Ih': Ih,
}
