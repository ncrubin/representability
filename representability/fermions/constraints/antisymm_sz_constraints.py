"""
Constraint generator and surrounding utilities for marginal problem in block diagonal
form by SZ operator.  antisymm_sz name comes from the fact that the alpha-alpha
and beta-beta blocks use antisymmetric basis functions as is common in the
2-RDM literature.  This is the most efficient non-redundant form for the SZ
symmetry adpating available.
"""
import sys
import numpy as np
from itertools import product
from representability.dualbasis import DualBasisElement, DualBasis


def gen_trans_2rdm(gem_dim, bas_dim):
    bas = dict(zip(range(gem_dim), product(range(1, bas_dim + 1),
                                           range(1, bas_dim + 1))))
    bas_rev = dict(zip(bas.values(), bas.keys()))

    D2ab_abas = {}
    D2ab_abas_rev = {}
    cnt = 0
    for xx in range(bas_dim):
        for yy in range(xx + 1, bas_dim):
            D2ab_abas[cnt] = (xx + 1, yy + 1)
            D2ab_abas_rev[(xx + 1, yy + 1)] = cnt
            cnt += 1

    D2ab_sbas = {}
    D2ab_sbas_rev = {}
    cnt = 0
    for xx in range(bas_dim):
        for yy in range(xx, bas_dim):
            D2ab_sbas[cnt] = (xx + 1, yy + 1)
            D2ab_sbas_rev[(xx + 1, yy + 1)] = cnt
            cnt += 1

    trans_mat = np.zeros((gem_dim, gem_dim))
    cnt = 0
    for xx in D2ab_abas.keys():
        i, j = D2ab_abas[xx]
        x1 = bas_rev[(i, j)]
        x2 = bas_rev[(j, i)]
        trans_mat[x1, cnt] = 1. / np.sqrt(2)
        trans_mat[x2, cnt] = -1. / np.sqrt(2)
        cnt += 1
    for xx in D2ab_sbas.keys():
        i, j = D2ab_sbas[xx]
        x1 = bas_rev[(i, j)]
        x2 = bas_rev[(j, i)]
        if x1 == x2:
            trans_mat[x1, cnt] = 1.0
        else:
            trans_mat[x1, cnt] = 1. / np.sqrt(2)
            trans_mat[x2, cnt] = 1. / np.sqrt(2)
        cnt += 1

    return trans_mat, D2ab_abas, D2ab_abas_rev, D2ab_sbas, D2ab_sbas_rev


def _trace_map(tname, dim, normalization):
    dbe = DualBasisElement()
    for i, j in product(range(dim), repeat=2):
        if i < j:
            dbe.add_element(tname, (i, j, i, j), 1.0)
    dbe.dual_scalar = normalization
    return dbe


def trace_d2_aa(dim, Na):
    dbe = DualBasisElement()
    for i, j in product(range(dim), repeat=2):
        if i < j:
            dbe.add_element('cckk_aa', (i, j, i, j), 2.0)
    dbe.dual_scalar = Na * (Na - 1)
    return dbe


def trace_d2_bb(dim, Nb):
    dbe = DualBasisElement()
    for i, j in product(range(dim), repeat=2):
        if i < j:
            dbe.add_element('cckk_bb', (i, j, i, j), 2.0)
    dbe.dual_scalar = Nb * (Nb - 1)
    return dbe


def trace_d2_ab(dim, Na, Nb):
    dbe = DualBasisElement()
    for i, j in product(range(dim), repeat=2):
        dbe.add_element('cckk_ab', (i, j, i, j), 1.0)
    dbe.dual_scalar = Na * Nb
    return dbe


def s_representability_d2ab(dim, N, M, S):
    """
    Constraint for S-representability

    PHYSICAL REVIEW A 72, 052505 2005


    :param dim: number of spatial basis functions
    :param N: Total number of electrons
    :param M: Sz expected value
    :param S: S(S + 1) is eigenvalue of S^{2}
    :return:
    """
    dbe = DualBasisElement()
    for i, j in product(range(dim), repeat=2):
        dbe.add_element('cckk_ab', (i, j, j, i), 1.0)
    dbe.dual_scalar = N/2.0 + M**2 - S*(S + 1)
    return dbe


def s_representability_d2ab_to_d2bb(dim):
    """
    Constraint the antisymmetric part of the alpha-beta matrix to be equal
    to the aa and bb components if a singlet

    :param dim:
    :return:
    """
    sma = dim * (dim - 1) // 2
    sms = dim * (dim + 1) // 2
    uadapt, d2ab_abas, d2ab_abas_rev, d2ab_sbas, d2ab_sbas_rev = \
        gen_trans_2rdm(dim**2, dim)

    d2ab_bas = {}
    d2aa_bas = {}
    cnt_ab = 0
    cnt_aa = 0
    for p, q in product(range(dim), repeat=2):
        d2ab_bas[(p, q)] = cnt_ab
        cnt_ab += 1
        if p < q:
            d2aa_bas[(p, q)] = cnt_aa
            cnt_aa += 1
    d2ab_rev = dict(zip(d2ab_bas.values(), d2ab_bas.keys()))
    d2aa_rev = dict(zip(d2aa_bas.values(), d2aa_bas.keys()))

    assert uadapt.shape == (int(dim)**2, int(dim)**2)
    dbe_list = []
    for r, s in product(range(dim * (dim - 1) // 2), repeat=2):
        if r < s:
            dbe = DualBasisElement()
            # lower triangle
            i, j = d2aa_rev[r]
            k, l = d2aa_rev[s]
            # aa element should equal the triplet block aa
            dbe.add_element('cckk_bb', (i, j, k, l), -0.5)
            coeff_mat = uadapt[:, [r]] @ uadapt[:, [s]].T
            for p, q in product(range(coeff_mat.shape[0]), repeat=2):
                if not np.isclose(coeff_mat[p, q], 0):
                    ii, jj = d2ab_rev[p]
                    kk, ll = d2ab_rev[q]
                    dbe.add_element('cckk_ab', (ii, jj, kk, ll), 0.5 * coeff_mat[p, q])

            # upper triangle .  Hermitian conjugate
            dbe.add_element('cckk_bb', (k, l, i, j), -0.5)
            coeff_mat = uadapt[:, [s]] @ uadapt[:, [r]].T
            for p, q in product(range(coeff_mat.shape[0]), repeat=2):
                if not np.isclose(coeff_mat[p, q], 0):
                    ii, jj = d2ab_rev[p]
                    kk, ll = d2ab_rev[q]
                    dbe.add_element('cckk_ab', (ii, jj, kk, ll),
                                    0.5 * coeff_mat[p, q])
            dbe.simplify()
            dbe_list.append(dbe)

        elif r == s:
           i, j = d2aa_rev[r]
           k, l = d2aa_rev[s]
           dbe = DualBasisElement()
           # aa element should equal the triplet block aa
           dbe.add_element('cckk_bb', (i, j, k, l), -1.0)
           coeff_mat = uadapt[:, [r]] @ uadapt[:, [s]].T
           for p, q in product(range(coeff_mat.shape[0]), repeat=2):
               if not np.isclose(coeff_mat[p, q], 0):
                   ii, jj = d2ab_rev[p]
                   kk, ll = d2ab_rev[q]
                   dbe.add_element('cckk_ab', (ii, jj, kk, ll), coeff_mat[p, q])
           dbe.simplify()
           dbe_list.append(dbe)

    return DualBasis(elements=dbe_list)


def s_representability_d2ab_to_d2aa(dim):
    """
    Constraint the antisymmetric part of the alpha-beta matrix to be equal
    to the aa and bb components if a singlet

    :param dim:
    :return:
    """
    sma = dim * (dim - 1) // 2
    sms = dim * (dim + 1) // 2
    uadapt, d2ab_abas, d2ab_abas_rev, d2ab_sbas, d2ab_sbas_rev = \
        gen_trans_2rdm(dim**2, dim)

    d2ab_bas = {}
    d2aa_bas = {}
    cnt_ab = 0
    cnt_aa = 0
    for p, q in product(range(dim), repeat=2):
        d2ab_bas[(p, q)] = cnt_ab
        cnt_ab += 1
        if p < q:
            d2aa_bas[(p, q)] = cnt_aa
            cnt_aa += 1
    d2ab_rev = dict(zip(d2ab_bas.values(), d2ab_bas.keys()))
    d2aa_rev = dict(zip(d2aa_bas.values(), d2aa_bas.keys()))

    assert uadapt.shape == (int(dim)**2, int(dim)**2)
    dbe_list = []
    for r, s in product(range(dim * (dim - 1) // 2), repeat=2):
        if r < s:
            dbe = DualBasisElement()
            # lower triangle
            i, j = d2aa_rev[r]
            k, l = d2aa_rev[s]
            # aa element should equal the triplet block aa
            dbe.add_element('cckk_aa', (i, j, k, l), -0.5)
            coeff_mat = uadapt[:, [r]] @ uadapt[:, [s]].T
            for p, q in product(range(coeff_mat.shape[0]), repeat=2):
                if not np.isclose(coeff_mat[p, q], 0):
                    ii, jj = d2ab_rev[p]
                    kk, ll = d2ab_rev[q]
                    dbe.add_element('cckk_ab', (ii, jj, kk, ll), 0.5 * coeff_mat[p, q])

            # upper triangle .  Hermitian conjugate
            dbe.add_element('cckk_aa', (k, l, i, j), -0.5)
            coeff_mat = uadapt[:, [s]] @ uadapt[:, [r]].T
            for p, q in product(range(coeff_mat.shape[0]), repeat=2):
                if not np.isclose(coeff_mat[p, q], 0):
                    ii, jj = d2ab_rev[p]
                    kk, ll = d2ab_rev[q]
                    dbe.add_element('cckk_ab', (ii, jj, kk, ll),
                                    0.5 * coeff_mat[p, q])
            dbe.simplify()
            dbe_list.append(dbe)

        elif r == s:
           i, j = d2aa_rev[r]
           k, l = d2aa_rev[s]
           dbe = DualBasisElement()
           # aa element should equal the triplet block aa
           dbe.add_element('cckk_aa', (i, j, k, l), -1.0)
           coeff_mat = uadapt[:, [r]] @ uadapt[:, [s]].T
           for p, q in product(range(coeff_mat.shape[0]), repeat=2):
               if not np.isclose(coeff_mat[p, q], 0):
                   ii, jj = d2ab_rev[p]
                   kk, ll = d2ab_rev[q]
                   dbe.add_element('cckk_ab', (ii, jj, kk, ll), coeff_mat[p, q])
           dbe.simplify()
           dbe_list.append(dbe)

    return DualBasis(elements=dbe_list)



def sz_representability(dim, M):
    """
    Constraint for S_z-representability

    Helgaker, Jorgensen, Olsen. Sz is one-body RDM constraint

    :param dim: number of spatial basis functions
    :param M: Sz expected value
    :return:
    """
    dbe = DualBasisElement()
    for i in range(dim):
        dbe.add_element('ck_a', (i, i), 0.5)
        dbe.add_element('ck_b', (i, i), -0.5)
    dbe.dual_scalar = M
    return dbe


def d2ab_d1a_mapping(dim, Nb):
    """
    Map the d2_spin-adapted 2-RDM to the D1 rdm

    :param Nb: number of beta electrons
    :param dim:
    :return:
    """
    return _contraction_base('cckk_ab', 'ck_a', dim, Nb, 0)


def d2ab_d1b_mapping(dim, Na):
    """
    Map the d2_spin-adapted 2-RDM to the D1 rdm

    :param Nb: number of beta electrons
    :param dim:
    :return:
    """
    db = DualBasis()
    for i in range(dim):
        for j in range(i, dim):
            dbe = DualBasisElement()
            for r in range(dim):
                dbe.add_element('cckk_ab', (r, i, r, j), 0.5)
                dbe.add_element('cckk_ab', (r, j, r, i), 0.5)

            dbe.add_element('ck_b', (i, j), -0.5 * Na)
            dbe.add_element('ck_b', (j, i), -0.5 * Na)
            dbe.dual_scalar = 0

            # dbe.simplify()
            db += dbe

    return db


def d2aa_d1a_mapping(dim, Na):
    """
    Map the d2_spin-adapted 2-RDM to the D1 rdm

    :param Nb: number of beta electrons
    :param dim:
    :return:
    """
    db = DualBasis()
    for i in range(dim):
        for j in range(i, dim):
            dbe = DualBasisElement()
            for r in range(dim):
                # Not in the basis because always zero
                if i == r or j == r:
                    continue
                else:
                    sir = 1 if i < r else -1
                    sjr = 1 if j < r else -1
                    ir_pair = (i, r) if i < r else (r, i)
                    jr_pair = (j, r) if j < r else (r, j)
                    if i == j:
                        dbe.add_element('cckk_aa', (ir_pair[0], ir_pair[1], jr_pair[0], jr_pair[1]), sir * sjr)
                    else:
                        dbe.add_element('cckk_aa', (ir_pair[0], ir_pair[1], jr_pair[0], jr_pair[1]), sir * sjr * 0.5)
                        dbe.add_element('cckk_aa', (jr_pair[0], jr_pair[1], ir_pair[0], ir_pair[1]), sir * sjr * 0.5)

            dbe.add_element('ck_a', (i, j), -0.5 * (Na - 1))
            dbe.add_element('ck_a', (j, i), -0.5 * (Na - 1))
            dbe.dual_scalar = 0

            # dbe.simplify()
            db += dbe

    return db


def d2bb_d1b_mapping(dim, Nb):
    """
    Map the d2_spin-adapted 2-RDM to the D1 rdm

    :param Nb: number of beta electrons
    :param dim:
    :return:
    """
    db = DualBasis()
    for i in range(dim):
        for j in range(i, dim):
            dbe = DualBasisElement()
            for r in range(dim):
                # Not in the basis because always zero
                if i == r or j == r:
                    continue
                else:
                    sir = 1 if i < r else -1
                    sjr = 1 if j < r else -1
                    ir_pair = (i, r) if i < r else (r, i)
                    jr_pair = (j, r) if j < r else (r, j)
                    if i == j:
                        dbe.add_element('cckk_bb', (ir_pair[0], ir_pair[1], jr_pair[0], jr_pair[1]), sir * sjr)
                    else:
                        dbe.add_element('cckk_bb', (ir_pair[0], ir_pair[1], jr_pair[0], jr_pair[1]), sir * sjr * 0.5)
                        dbe.add_element('cckk_bb', (jr_pair[0], jr_pair[1], ir_pair[0], ir_pair[1]), sir * sjr * 0.5)

            dbe.add_element('ck_b', (i, j), -0.5 * (Nb - 1))
            dbe.add_element('ck_b', (j, i), -0.5 * (Nb - 1))
            dbe.dual_scalar = 0

            # dbe.simplify()
            db += dbe

    return db
# TODO: Implement trace constraints on these spin-blocks (PHYSICAL REVIEW A 72, 052505 2005,...
# E. Perez-Romero, L. M. Tel, and C. Valdemoro, Int. J. Quantum Chem. 61, 55 19970)
# def d2aa_d1a_mapping(Na, dim):
#     """
#     Map the d2_spin-adapted 2-RDM to the D1 rdm
#
#     :param Nb: number of beta electrons
#     :param dim:
#     :return:
#     """
#     return _contraction_base('cckk_aa', 'ck_a', dim, Na - 1, 1)
#
#
# def d2bb_d1b_mapping(Nb, dim):
#     """
#     Map the d2_spin-adapted 2-RDM to the D1 rdm
#
#     :param Nb: number of beta electrons
#     :param dim:
#     :return:
#     """
#     return _contraction_base('cckk_bb', 'ck_b', dim, Nb - 1, 1)


def _contraction_base(tname_d2, tname_d1, dim, normalization, offset):
    db = DualBasis()
    for i in range(dim):
        for j in range(i + offset, dim):
            dbe = DualBasisElement()
            for r in range(dim):
                dbe.add_element(tname_d2, (i, r, j, r), 0.5)
                dbe.add_element(tname_d2, (j, r, i, r), 0.5)

            dbe.add_element(tname_d1, (i, j), -0.5 * normalization)
            dbe.add_element(tname_d1, (j, i), -0.5 * normalization)
            dbe.dual_scalar = 0

            # dbe.simplify()
            db += dbe

    return db


def _d1_q1_mapping(tname_d1, tname_q1, dim):
    db = DualBasis()
    for i in range(dim):
        for j in range(i, dim):
            dbe = DualBasisElement()
            if i != j:
                dbe.add_element(tname_d1, (i, j), 0.5)
                dbe.add_element(tname_d1, (j, i), 0.5)
                dbe.add_element(tname_q1, (i, j), 0.5)
                dbe.add_element(tname_q1, (j, i), 0.5)
                dbe.dual_scalar = 0.0
            else:
                dbe.add_element(tname_d1, (i, j), 1.0)
                dbe.add_element(tname_q1, (i, j), 1.0)
                dbe.dual_scalar = 1.0

            db += dbe# .simplify()

    return db


def d1a_d1b_mapping(tname_d1a, tname_d1b, dim):
    db = DualBasis()
    for i in range(dim):
        for j in range(i, dim):
            dbe = DualBasisElement()
            if i != j:
                dbe.add_element(tname_d1a, (i, j), 0.5)
                dbe.add_element(tname_d1a, (j, i), 0.5)
                dbe.add_element(tname_d1b, (i, j), -0.5)
                dbe.add_element(tname_d1b, (j, i), -0.5)
                dbe.dual_scalar = 0.0
            else:
                dbe.add_element(tname_d1a, (i, j), 1.0)
                dbe.add_element(tname_d1b, (i, j), -1.0)
                dbe.dual_scalar = 0.0

            db += dbe# .simplify()

    return db


def d1a_q1a_mapping(dim):
    """
    Generate the dual basis elements for spin-blocks of d1 and q1

    :param dim: spatial basis rank
    :return:
    """
    return _d1_q1_mapping('ck_a', 'kc_a', dim)


def d1b_q1b_mapping(dim):
    """
    Generate the dual basis elements for spin-blocks of d1 and q1

    :param dim: spatial basis rank
    :return:
    """
    return _d1_q1_mapping('ck_b', 'kc_b', dim)


# TODO: Modularize the spin-block constraints for Q2
def d2_q2_mapping(dim):
    """
    Map each d2 block to the q2 block

    :param dim: rank of spatial single-particle basis
    :return:
    """
    krond = np.eye(dim)
    def d2q2element(p, q, r, s, factor, tname_d1_1, tname_d2, tname_q2): # , spin_string):
        """
        # (   1.00000) cre(r) cre(s) des(q) des(p)

        # (   1.00000) kdelta(p,s) cre(r) des(q)

        # (  -1.00000) kdelta(p,r) cre(s) des(q)

        # (  -1.00000) kdelta(q,s) cre(r) des(p)

        # (   1.00000) kdelta(q,r) cre(s) des(p)

        # (  -1.00000) kdelta(p,s) kdelta(q,r)
        # (   1.00000) kdelta(p,r) kdelta(q,s)
        """
        dbe = DualBasisElement()
        dbe.add_element(tname_q2, (p, q, r, s), -factor)
        dbe.add_element(tname_d2, (r, s, p, q), factor)

        if p == s:
            dbe.add_element(tname_d1_1, (r, q), factor)
        if p == r:
            dbe.add_element(tname_d1_1, (s, q), -factor)
        if q == s:
            dbe.add_element(tname_d1_1, (r, p), -factor)
        if q == r:
            dbe.add_element(tname_d1_1, (s, p), factor)

        # remember the negative sign because AX = b
        dbe.dual_scalar = -factor * (krond[p, r] * krond[q, s] - krond[p, s] * krond[q, r])

        # dbe.add_element('kkcc_' + spin_string + spin_string, (r, s, p, q), factor)
        # if q == s:
        #     dbe.add_element('ck_' + spin_string, (p, r), factor)
        # if p == s:
        #     dbe.add_element('ck_' + spin_string, (q, r), -factor)
        # if q == r:
        #     dbe.add_element('kc_' + spin_string, (s, p), factor)
        # if p == r:
        #     dbe.add_element('kc_' + spin_string, (s, q), -factor)

        # dbe.add_element('cckk_' + spin_string + spin_string, (p, q, r, s), -factor)
        # dbe.dual_scalar = 0
        return dbe

    def d2q2element_ab(p, q, r, s, factor):
        dbe = DualBasisElement()
        if q == s:
            dbe.add_element('ck_a', (p, r), factor)
        if p == r:
            dbe.add_element('kc_b', (s, q), -factor)

        dbe.add_element('kkcc_ab', (r, s, p, q), factor)
        dbe.add_element('cckk_ab', (p, q, r, s), -factor)
        # dbe.dual_scalar = -krond[q, s]*krond[p, r] * factor
        dbe.dual_scalar = 0
        return dbe

    dual_basis_list = []
    # for p, q, r, s in product(range(dim), repeat=4):
    from representability.tensor import index_tuple_basis
    gem_aa = []
    gem_ab = []
    for p, q in product(range(dim), repeat=2):
        if p < q:
            gem_aa.append((p, q))
        gem_ab.append((p, q))

    bas_aa = index_tuple_basis(gem_aa)
    bas_ab = index_tuple_basis(gem_ab)

    for i, j in product(range(dim * (dim - 1)//2), repeat=2):
        if i >= j:
            p, q = bas_aa.fwd(i)
            r, s  = bas_aa.fwd(j)
        # if p < q and r < s and p * dim + q < r * dim + s:
            dbe1 = d2q2element(p, q, r, s, 0.5, 'ck_a', 'cckk_aa', 'kkcc_aa')
            dbe2 = d2q2element(r, s, p, q, 0.5, 'ck_a', 'cckk_aa', 'kkcc_aa')
            # dbe1 = d2q2element(p, q, r, s, 0.5, 'a')
            # dbe2 = d2q2element(r, s, p, q, 0.5, 'a')
            dual_basis_list.append(dbe1.join_elements(dbe2))

            # dbe1 = d2q2element(p, q, r, s, 0.5, 'b')
            # dbe2 = d2q2element(r, s, p, q, 0.5, 'b')
            dbe1 = d2q2element(p, q, r, s, 0.5, 'ck_b', 'cckk_bb', 'kkcc_bb')
            dbe2 = d2q2element(r, s, p, q, 0.5, 'ck_b', 'cckk_bb', 'kkcc_bb')
            dual_basis_list.append(dbe1.join_elements(dbe2))

        # # if p < q and r < s and p == r and q == s:
        #     # dbe1 = d2q2element(p, q, r, s, 1., 'a')
        #     dbe = d2q2element(p, q, r, s, 1., 'ck_a', 'cckk_aa', 'kkcc_aa')
        #     dual_basis_list.append(dbe)

        #     # # dbe1 = d2q2element(p, q, r, s, 1., 'b')
        #     dbe = d2q2element(p, q, r, s, 1., 'ck_b', 'cckk_bb', 'kkcc_bb')
        #     dual_basis_list.append(dbe)

    for i, j in product(range(dim * dim), repeat=2):
        if i >= j:
            p, q = bas_ab.fwd(i)
            r, s = bas_ab.fwd(j)
        # if p * dim + q <= r * dim + s:
            dbe1 = d2q2element_ab(p, q, r, s, 0.5)
            dbe2 = d2q2element_ab(r, s, p, q, 0.5)
            dual_basis_list.append(dbe1.join_elements(dbe2))

    return DualBasis(elements=dual_basis_list)


# TODO: Modularize the spin-block constraints for G2
def d2_g2_mapping(dim):
    """
    Map each d2 blcok to the g2 blocks

    :param dim: rank of spatial single-particle basis
    :return:
    """
    krond = np.eye(dim)
    # d2 -> g2

    def g2d2map_aa_or_bb(p, q, r, s, dim, key, factor=1.0):
        """
        Accept pqrs of G2 and map to D2
        """
        dbe = DualBasisElement()
        quad = {'aa': [0, 0], 'bb': [1, 1]}
        dbe.add_element('ckck_aabb', (p * dim + q + quad[key][0]*dim**2, r * dim + s + quad[key][1]*dim**2),
                        -1.0 * factor)
        if q == s:
            dbe.add_element('ck_' + key[0], (p, r), krond[q, s] * factor)
        if p != s and r != q:
            gem1 = tuple(sorted([p, s]))
            gem2 = tuple(sorted([r, q]))
            parity = (-1)**(p < s) * (-1)**(r < q)
            # factor of 0.5 is from the spin-adapting
            dbe.add_element('cckk_' + key, (gem1[0], gem1[1], gem2[0], gem2[1]), parity * -factor)

        dbe.dual_scalar = 0
        return dbe

    def g2d2map_aabb(p, q, r, s, dim, key, factor=1.0):
        """
        Accept pqrs of G2 and map to D2
        """
        dbe = DualBasisElement()
        # this is ugly.  :(
        quad = {'aabb': [0, 1], 'bbaa': [1, 0]}
        dbe.add_element('ckck_aabb', (p * dim + q + quad[key][0]*dim**2, r * dim + s + quad[key][1]*dim**2),
                        1.0 * factor)
        dbe.add_element('ckck_aabb', (r * dim + s + quad[key[::-1]][0]*dim**2, p * dim + q + quad[key[::-1]][1]*dim**2),
                        1.0 * factor)

        dbe.add_element('cckk_ab', (p, s, q, r), -1.0 * factor)
        dbe.add_element('cckk_ab', (q, r, p, s), -1.0 * factor)
        dbe.dual_scalar = 0.0
        return dbe

    def g2d2map_ab(p, q, r, s, key, factor=1.0):
        dbe = DualBasisElement()
        if key == 'ab':
            if q == s:
                dbe.add_element('ck_' + key[0], (p, r), krond[q, s] * factor)

            dbe.add_element('cckk_' + key, (p, s, r, q), -1.0 * factor)
        elif key == 'ba':
            if q == s:
                dbe.add_element('ck_' + key[0], (p, r), krond[q, s] * factor)

            dbe.add_element('cckk_ab', (s, p, q, r), -1.0 * factor)
        else:
            raise TypeError("I only accept ab or ba blocks")

        dbe.add_element('ckck_' + key, (p, q, r, s), -1.0 * factor)
        dbe.dual_scalar = 0.0
        return dbe

    dual_basis_list = []
    # do aa_aa block then bb_block of the superblock
    for key in ['bb', 'aa']:
        for p, q, r, s in product(range(dim), repeat=4):
            if p * dim + q <= r * dim + s:
                dbe_1 = g2d2map_aa_or_bb(p, q, r, s, dim, key, factor=0.5)
                dbe_2 = g2d2map_aa_or_bb(r, s, p, q, dim, key, factor=0.5)
                dual_basis_list.append(dbe_1.join_elements(dbe_2))

    # this constraint is over the entire block!
    for key in ['aabb']:
        for p, q, r, s in product(range(dim), repeat=4):
            dbe = g2d2map_aabb(p, q, r, s, dim, key, factor=1.0)
            # db += dbe
            dual_basis_list.append(dbe)

    # # ab ba blocks of G2
    for key in ['ab', 'ba']:
        for p, q, r, s in product(range(dim), repeat=4):
            if p * dim + q <= r * dim + s:
                dbe_1 = g2d2map_ab(p, q, r, s, key, factor=0.5)
                dbe_2 = g2d2map_ab(r, s, p, q, key, factor=0.5)
                dual_basis_list.append(dbe_1.join_elements(dbe_2))

    return DualBasis(elements=dual_basis_list)


def d2_e2_mapping(dim, bas_aa, bas_ab, measured_tpdm_aa, measured_tpdm_bb, measured_tpdm_ab):
    """
    Generate constraints such that the error matrix and the d2 matrices look like the measured matrices

    :param dim: spatial basis dimension
    :param measured_tpdm_aa: two-marginal of alpha-alpha spins
    :param measured_tpdm_bb: two-marginal of beta-beta spins
    :param measured_tpdm_ab: two-marginal of alpha-beta spins
    :return:
    """
    db = DualBasis()
    # first constrain the aa-matrix
    aa_dim = dim * (dim - 1) / 2
    ab_dim = dim **2

    # map the aa matrix to the measured_tpdm_aa
    for p, q, r, s in product(range(dim), repeat=4):
        if p < q and r < s and bas_aa[(p, q)] <= bas_aa[(r, s)]:
            dbe = DualBasisElement()

            # two elements of D2aa
            dbe.add_element('cckk_aa', (p, q, r, s), 0.5)
            dbe.add_element('cckk_aa', (r, s, p, q), 0.5)

            # four elements of the E2aa
            dbe.add_element('cckk_me_aa', (bas_aa[(p, q)] + aa_dim, bas_aa[(r, s)]), 0.25)
            dbe.add_element('cckk_me_aa', (bas_aa[(r, s)] + aa_dim, bas_aa[(p, q)]), 0.25)
            dbe.add_element('cckk_me_aa', (bas_aa[(p, q)], bas_aa[(r, s)] + aa_dim), 0.25)
            dbe.add_element('cckk_me_aa', (bas_aa[(r, s)], bas_aa[(p, q)] + aa_dim), 0.25)

            dbe.dual_scalar = measured_tpdm_aa[bas_aa[(p, q)], bas_aa[(r, s)]].real
            dbe.simplify()

            # construct the dbe for constraining the [0, 0] orthant to the idenity matrix
            dbe_identity_aa = DualBasisElement()
            if bas_aa[(p, q)] == bas_aa[(r, s)]:
                dbe_identity_aa.add_element('cckk_me_aa', (bas_aa[(p, q)], bas_aa[(r, s)]), 1.0)
                dbe_identity_aa.dual_scalar = 1.0
            else:
                dbe_identity_aa.add_element('cckk_me_aa', (bas_aa[(p, q)], bas_aa[(r, s)]), 0.5)
                dbe_identity_aa.add_element('cckk_me_aa', (bas_aa[(r, s)], bas_aa[(p, q)]), 0.5)
                dbe_identity_aa.dual_scalar = 0.0

            db += dbe
            db += dbe_identity_aa

    # map the bb matrix to the measured_tpdm_bb
    for p, q, r, s in product(range(dim), repeat=4):
        if p < q and r < s and bas_aa[(p, q)] <= bas_aa[(r, s)]:
            dbe = DualBasisElement()

            # two elements of D2bb
            dbe.add_element('cckk_bb', (p, q, r, s), 0.5)
            dbe.add_element('cckk_bb', (r, s, p, q), 0.5)

            # four elements of the E2bb
            dbe.add_element('cckk_me_bb', (bas_aa[(p, q)] + aa_dim, bas_aa[(r, s)]), 0.25)
            dbe.add_element('cckk_me_bb', (bas_aa[(r, s)] + aa_dim, bas_aa[(p, q)]), 0.25)
            dbe.add_element('cckk_me_bb', (bas_aa[(p, q)], bas_aa[(r, s)] + aa_dim), 0.25)
            dbe.add_element('cckk_me_bb', (bas_aa[(r, s)], bas_aa[(p, q)] + aa_dim), 0.25)

            dbe.dual_scalar = measured_tpdm_bb[bas_aa[(p, q)], bas_aa[(r, s)]].real
            dbe.simplify()

            # construct the dbe for constraining the [0, 0] orthant to the idenity matrix
            dbe_identity_bb = DualBasisElement()
            if bas_aa[(p, q)] == bas_aa[(r, s)]:
                dbe_identity_bb.add_element('cckk_me_bb', (bas_aa[(p, q)], bas_aa[(r, s)]), 1.0)
                dbe_identity_bb.dual_scalar = 1.0
            else:
                dbe_identity_bb.add_element('cckk_me_bb', (bas_aa[(p, q)], bas_aa[(r, s)]), 0.5)
                dbe_identity_bb.add_element('cckk_me_bb', (bas_aa[(r, s)], bas_aa[(p, q)]), 0.5)
                dbe_identity_bb.dual_scalar = 0.0

            db += dbe
            db += dbe_identity_bb

    # map the ab matrix to the measured_tpdm_ab
    for p, q, r, s in product(range(dim), repeat=4):
        if bas_ab[(p, q)] <= bas_ab[(r, s)]:
            dbe = DualBasisElement()

            # two elements of D2ab
            dbe.add_element('cckk_ab', (p, q, r, s), 0.5)
            dbe.add_element('cckk_ab', (r, s, p, q), 0.5)

            # four elements of the E2ab
            dbe.add_element('cckk_me_ab', (bas_ab[(p, q)] + ab_dim, bas_ab[(r, s)]), 0.25)
            dbe.add_element('cckk_me_ab', (bas_ab[(r, s)] + ab_dim, bas_ab[(p, q)]), 0.25)
            dbe.add_element('cckk_me_ab', (bas_ab[(p, q)], bas_ab[(r, s)] + ab_dim), 0.25)
            dbe.add_element('cckk_me_ab', (bas_ab[(r, s)], bas_ab[(p, q)] + ab_dim), 0.25)

            dbe.dual_scalar = measured_tpdm_ab[bas_ab[(p, q)], bas_ab[(r, s)]].real
            dbe.simplify()

            # construct the dbe for constraining the [0, 0] orthant to the idenity matrix
            dbe_identity_ab = DualBasisElement()
            if bas_ab[(p, q)] == bas_ab[(r, s)]:
                dbe_identity_ab.add_element('cckk_me_ab', (bas_ab[(p, q)], bas_ab[(r, s)]), 1.0)
                dbe_identity_ab.dual_scalar = 1.0
            else:
                dbe_identity_ab.add_element('cckk_me_ab', (bas_ab[(p, q)], bas_ab[(r, s)]), 0.5)
                dbe_identity_ab.add_element('cckk_me_ab', (bas_ab[(r, s)], bas_ab[(p, q)]), 0.5)
                dbe_identity_ab.dual_scalar = 0.0

            db += dbe
            db += dbe_identity_ab

    return db


def sz_adapted_linear_constraints(dim, Na, Nb, constraint_list, S=0, M=0):
    """
    Generate the dual basis for the v2-RDM program

    :param dim: rank of the spatial single-particle basis
    :param Na: Number of alpha electrons
    :param Nb: Number of beta electrons
    :param constraint_list:  List of strings indicating which constraints to make
    :return:
    """
    if Na != Nb and M == 0:
        raise TypeError("you gave me impossible quantum numbers")

    dual_basis = DualBasis()
    if 'cckk' in constraint_list:
        dual_basis += trace_d2_ab(dim, Na, Nb)
        dual_basis += s_representability_d2ab(dim, Na + Nb, M, S)

    # Including these would introduce linear independence.  Why?
        dual_basis += trace_d2_aa(dim, Na)
        dual_basis += trace_d2_bb(dim, Nb)

        if Na == Nb:
            dual_basis += s_representability_d2ab_to_d2aa(dim)
            dual_basis += s_representability_d2ab_to_d2bb(dim)

    if 'ck' in constraint_list:
        if Na > 1:
            dual_basis += d2aa_d1a_mapping(dim, Na)
            dual_basis += trace_d2_aa(dim, Na)
        else:
            dual_basis += trace_d2_aa(dim, Na)
        if Nb > 1:
            dual_basis += d2bb_d1b_mapping(dim, Nb)
            dual_basis += trace_d2_bb(dim, Nb)
        else:
            dual_basis += trace_d2_bb(dim, Nb)

        dual_basis += d2ab_d1b_mapping(dim, Na)
        dual_basis += d2ab_d1a_mapping(dim, Nb)

        dual_basis += d1a_q1a_mapping(dim)
        dual_basis += d1b_q1b_mapping(dim)

        # dual_basis += d1a_d1b_mapping('ck_a', 'ck_b', dim)

        # this might not be needed if s_representability is enforced
        if Na + Nb > 2:
            dual_basis += sz_representability(dim, M)

    if 'kkcc' in constraint_list:
        dual_basis += d2_q2_mapping(dim)

    if 'ckck' in constraint_list:
        dual_basis += d2_g2_mapping(dim)

    return dual_basis
