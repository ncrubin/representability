"""
Utilities for returning marginals with SZ spin-adapting.
The alpha-alpha basis functions are symmetric basis functions.
This is the spin adapting that appears in most of the 2-RDM papers.

More details can be found in PhysRevA.72.052505

The 1-RDMs are block diagonal matrices corresponding to their spin-index
(alpha or beta).  The 2-RDM and 2-Hole-RDM contain three blocks
aa, bb, ab blocks with (r*(r - 1)/2), r*(r - 1)/2, and r**2) linear size
where r is the spatial basis function rank.
"""
from typing import Union
import numpy as np
import scipy as sp
from itertools import product
from representability.fermions.density.antisymm_sz_maps import (
    map_d2anti_d1_sz, map_d2symm_d1_sz, get_sz_spin_adapted, )
import openfermion as of



class AntiSymmOrbitalDensity:

    def __init__(self, rho, dim): # , transform=JWTransform()):
        """
        Full Sz symmetry adapting

        The internal `_tensor_construct()` method is inherited from the
        `SymmOrbitalDensity` object.  The methods `construct_opdm()` and
        `construc_ohdm()` are also inherited as their behavior is the same.

        The full symmetry adapting requires a different structure and data
        abstraction for the marginals.

        :param rho: N-qubit density matrix
        :param dim: single particle basis rank (spin-orbital rank)
        :param transform: Fermionc to qubit transform object
        """
        # super(AntiSymmOrbitalDensity, self).__init__(rho, dim)
        # self.transform = transform
        self.dim = dim
        self.rho = rho

    def get_tpdm(self, wfn: Union[sp.sparse.coo_matrix, sp.sparse.csr_matrix,
                            sp.sparse.csc_matrix], nso: int) -> np.ndarray:
        """
        :param wfn: wavefunction as coo_sparse matrix
        :param nso:  number-spin-orbitals
        """
        if isinstance(wfn, (sp.sparse.coo_matrix, sp.sparse.csc_matrix)):
            wfn = wfn.tocsr()
            # wfn = wfn.toarray()
        tpdm = np.zeros((nso, nso, nso, nso), dtype=wfn.dtype)
        creation_ops = [
            of.get_sparse_operator(
                of.jordan_wigner(of.FermionOperator(((p, 1)))),
                n_qubits=nso)
            for p in range(nso)
        ]
        for p, q, r, s in product(range(nso), repeat=4):
            if p == q or r == s:
                continue
            operator = creation_ops[p] @ creation_ops[q] @ creation_ops[
                r].conj().transpose() @ creation_ops[s].conj().transpose()
            if wfn.shape[0] == wfn.shape[
                1]:  # we are working with a density matrix
                val = np.sum((wfn @ operator).diagonal())
            else:
                val = wfn.conj().transpose() @ operator @ wfn
            if isinstance(val, (float, complex, int)):
                tpdm[p, q, r, s] = val
            else:
                tpdm[p, q, r, s] = val[0, 0]
        return tpdm

    def get_opdm(self, wfn: Union[sp.sparse.coo_matrix, sp.sparse.csr_matrix,
                            sp.sparse.csc_matrix], nso: int) -> np.ndarray:
        if isinstance(wfn, (sp.sparse.coo_matrix, sp.sparse.csc_matrix)):
            wfn = wfn.tocsr()
            wfn = wfn.toarray()
        opdm = np.zeros((nso, nso), dtype=wfn.dtype)
        a_ops = [
            of.get_sparse_operator(
                of.jordan_wigner(of.FermionOperator(((p, 0)))),
                n_qubits=nso)
            for p in range(nso)
        ]
        for p, q in product(range(nso), repeat=2):
            operator = a_ops[p].conj().T @ a_ops[q]
            if wfn.shape[0] == wfn.shape[
                1]:  # we are working with a density matrix
                val = np.trace(wfn @ operator)
            else:
                val = wfn.conj().transpose() @ operator @ wfn
            # print((p, q, r, s), val.toarray()[0, 0])
            if isinstance(val, (float, complex, int)):
                opdm[p, q] = val
            else:
                opdm[p, q] = val[0, 0]

            opdm[p, q] = val
        return opdm

    def get_tqdm(self, wfn: Union[sp.sparse.coo_matrix, sp.sparse.csr_matrix,
                            sp.sparse.csc_matrix], nso: int) -> np.ndarray:
        """

        :param wfn: wavefunction as coo_sparse matrix
        :param nso:  number-spin-orbitals
        """
        if isinstance(wfn, (sp.sparse.coo_matrix, sp.sparse.csc_matrix)):
            wfn = wfn.tocsr()
            wfn = wfn.toarray()
        tpdm = np.zeros((nso, nso, nso, nso), dtype=wfn.dtype)
        a_ops = [
            of.get_sparse_operator(
                of.jordan_wigner(of.FermionOperator(((p, 0)))),
                n_qubits=nso)
            for p in range(nso)
        ]
        for p, q, r, s in product(range(nso), repeat=4):
            if p == q or r == s:
                continue
            operator = a_ops[p] @ a_ops[q] @ a_ops[r].conj().transpose() @ \
                       a_ops[s].conj().transpose()
            if wfn.shape[0] == wfn.shape[
                1]:  # we are working with a density matrix
                val = np.trace(wfn @ operator)
            else:
                val = wfn.conj().transpose() @ operator @ wfn
            # print((p, q, r, s), val.toarray()[0, 0])
            tpdm[p, q, r, s] = val  # val.toarray()[0, 0]
        return tpdm

    def get_phdm(self, wfn: Union[sp.sparse.coo_matrix, sp.sparse.csr_matrix,
                            sp.sparse.csc_matrix], nso: int) -> np.ndarray:
        """

        :param wfn: wavefunction as coo_sparse matrix
        :param nso:  number-spin-orbitals
        """
        if isinstance(wfn, (sp.sparse.coo_matrix, sp.sparse.csc_matrix)):
            wfn = wfn.tocsr()
            wfn = wfn.toarray()
        tpdm = np.zeros((nso, nso, nso, nso), dtype=wfn.dtype)
        a_ops = [
            of.get_sparse_operator(
                of.jordan_wigner(of.FermionOperator(((p, 0)))),
                n_qubits=nso)
            for p in range(nso)
        ]
        for p, q, r, s in product(range(nso), repeat=4):
            operator = a_ops[p].conj().T @ a_ops[q] @ a_ops[r].conj().T @ a_ops[s]
            if wfn.shape[0] == wfn.shape[
                1]:  # we are working with a density matrix
                val = np.trace(wfn @ operator)
            else:
                val = wfn.conj().transpose() @ operator @ wfn
            # print((p, q, r, s), val.toarray()[0, 0])
            tpdm[p, q, r, s] = val  # val.toarray()[0, 0]
        return tpdm

    def construct_opdm(self):
        """
        Return the two-particle density matrix
        <psi|a_{p, sigma}^{\dagger}a_{q, sigma'}^{\dagger}a_{s, sigma'}a_{r, sigma}|psi> """
        opdm = self.get_opdm(self.rho, self.dim)
        return opdm[::2, ::2], opdm[1::2, 1::2]

    def construct_ohdm(self):
        """
        Return the two-particle density matrix
        <psi|a_{p, sigma}^{\dagger}a_{q, sigma'}^{\dagger}a_{s, sigma'}a_{r, sigma}|psi> """
        opdm = self.get_opdm(self.rho, self.dim)
        ohdm = np.eye(opdm.shape[0]) - opdm
        return ohdm[::2, ::2], ohdm[1::2, 1::2]

    def construct_tpdm(self):
        """
        Return the two-particle density matrix
        <psi|a_{p, sigma}^{\dagger}a_{q, sigma'}^{\dagger}a_{s, sigma'}a_{r, sigma}|psi> """
        tpdm = self.get_tpdm(self.rho, self.dim)

        # build basis look up table
        bas_aa = {}
        bas_ab = {}
        cnt_aa = 0
        cnt_ab = 0
        sdim = self.dim // 2
        # iterate over spatial orbital indices
        for p, q in product(range(int(self.dim/2)), repeat=2):
            if p < q:
                bas_aa[(p, q)] = cnt_aa
                cnt_aa += 1
            bas_ab[(p, q)] = cnt_ab
            cnt_ab += 1

        rev_bas_aa = dict(zip(bas_aa.values(), bas_aa.keys()))
        rev_bas_ab = dict(zip(bas_ab.values(), bas_ab.keys()))

        d2_aa = np.zeros((sdim * (sdim - 1) // 2, sdim * (sdim - 1) // 2))
        d2_bb = np.zeros((sdim * (sdim - 1) // 2, sdim * (sdim - 1) // 2))
        d2_ab = np.zeros((sdim * sdim, sdim * sdim))

        for r, s in product(range(len(bas_aa)), repeat=2):
            i, j = rev_bas_aa[r]
            k, l = rev_bas_aa[s]
            d2_aa[r, s] = tpdm[2 * i, 2 * j, 2 * l, 2 * k].real
            d2_bb[r, s] = tpdm[2 * i + 1, 2 * j + 1, 2 * l + 1, 2 * k + 1].real
        for r, s in product(range(len(bas_ab)), repeat=2):
            i, j = rev_bas_ab[r]
            k, l = rev_bas_ab[s]
            d2_ab[r, s] = tpdm[2 * i, 2 * j + 1, 2 * l + 1, 2 * k].real

        return d2_aa, d2_bb, d2_ab, [bas_aa, bas_ab]

    def construct_thdm(self):
        """
        Return the two-hole density matrix

        <psi|a_{p, sigma}a_{q, sigma'}a_{s, sigma'}^{\dagger}a_{r, sigma}^{\dagger}|psi>
        """
        tqdm = self.get_tqdm(self.rho, self.dim)

        sdim = self.dim // 2
        # iterate over spatial orbital indices
        cnt_aa = 0
        cnt_ab = 0
        bas_aa = {}
        bas_ab = {}
        for p, q in product(range(int(self.dim / 2)), repeat=2):
            if p < q:
                bas_aa[(p, q)] = cnt_aa
                cnt_aa += 1
            bas_ab[(p, q)] = cnt_ab
            cnt_ab += 1

        rev_bas_aa = dict(zip(bas_aa.values(), bas_aa.keys()))
        rev_bas_ab = dict(zip(bas_ab.values(), bas_ab.keys()))

        q2_aa = np.zeros((sdim * (sdim - 1) // 2, sdim * (sdim - 1) // 2))
        q2_bb = np.zeros((sdim * (sdim - 1) // 2, sdim * (sdim - 1) // 2))
        q2_ab = np.zeros((sdim * sdim, sdim * sdim))

        for r, s in product(range(len(bas_aa)), repeat=2):
            i, j = rev_bas_aa[r]
            k, l = rev_bas_aa[s]
            q2_aa[r, s] = tqdm[2 * i, 2 * j, 2 * l, 2 * k].real
            q2_bb[r, s] = tqdm[2 * i + 1, 2 * j + 1, 2 * l + 1, 2 * k + 1].real
        for r, s in product(range(len(bas_ab)), repeat=2):
            i, j = rev_bas_ab[r]
            k, l = rev_bas_ab[s]
            q2_ab[r, s] = tqdm[2 * i, 2 * j + 1, 2 * l + 1, 2 * k].real

        return q2_aa, q2_bb, q2_ab, [bas_aa, bas_ab]


    def construct_phdm(self):
        """
        Return the particle-hole density matrix

        <psi|a_{p, sigma}^{\dagger}a_{q, sigma'}a_{s, sigma'}^{\dagger}a_{r, sigma}|psi>
        """
        conjugate = [-1, 1, -1, 1]
        spin_blocks = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1],
                       [0, 0, 1, 1], [1, 1, 0, 0]]
        phdm = self.get_phdm(self.rho, self.dim)
        sdim = self.dim//2
        rdms = []
        rdms.append(np.einsum('pqrs->pqsr', phdm[::2, 1::2, 1::2, ::2]).reshape((sdim**2, sdim**2)))
        rdms.append(np.einsum('pqrs->pqsr', phdm[1::2, ::2, ::2, 1::2]).reshape((sdim**2, sdim**2)))
        rdms.append(np.einsum('pqrs->pqsr', phdm[::2, ::2, ::2, ::2]))
        rdms.append(np.einsum('pqrs->pqsr', phdm[1::2, 1::2, 1::2, 1::2]))
        rdms.append(np.einsum('pqrs->pqsr', phdm[::2, ::2, 1::2, 1::2]))
        rdms.append(np.einsum('pqrs->pqsr', phdm[1::2, 1::2, ::2, ::2]))
        # unfortunately, this will not do in terms of a tensor structure.  Yes, the code works but when it is unrolled
        # into a matrix on the SDP side there is no guarantee of the correct ordering.

        # g2_aabb = np.zeros((2, 2, self.dim/2, self.dim/2,
        #                     self.dim/2, self.dim/2), dtype=complex)
        # g2_aabb[0, 0, :, :, :, :] = rdms[2]
        # g2_aabb[1, 1, :, :, :, :] = rdms[3]
        # g2_aabb[0, 1, :, :, :, :] = rdms[4]
        # g2_aabb[1, 0, :, :, :, :] = rdms[5]

        dim = int(self.dim / 2)
        mm = dim ** 2
        g2_aabb = np.zeros((2*mm, 2*mm))
        # unroll into the blocks
        for p, q, r, s in product(range(int(self.dim/2)), repeat=4):
            g2_aabb[p * dim + q, r * dim + s] = rdms[2][p, q, r, s].real
            g2_aabb[p * dim + q + dim**2, r * dim + s + dim**2] = rdms[3][p, q, r, s].real
            g2_aabb[p * dim + q, r * dim + s + dim**2] = rdms[4][p, q, r, s].real
            g2_aabb[p * dim + q + dim**2, r * dim + s] = rdms[5][p, q, r, s].real

        return [rdms[0], rdms[1], g2_aabb]


    def construct_tpdm_error_matrix(self, error_tpdm):
        """
        Construct the error matrix for a block of the marginal

        [I] [E]
        [E*] [0]

        where I is the identity matrix, E is ^{2}D_{meas} - ^{2}D_{var}, F is a matrix of Free variables

        :param corrupted_tpdm:
        :return:
        """
        if np.ndim(error_tpdm) != 2:
            raise TypeError("corrupted_tpdm needs to be a matrix")

        dim = int(error_tpdm.shape[0])
        top_row_emat = np.hstack((np.eye(dim), error_tpdm))
        bottom_row_emat = np.hstack((error_tpdm.T, np.zeros((dim, dim))))
        error_schmidt_matrix = np.vstack((top_row_emat, bottom_row_emat))

        return error_schmidt_matrix


def check_d2_d1_sz_antisymm(tpdm, opdm, normalization, bas):
    """
    check the contractive map from d2 to d1
    """
    opdm_test = map_d2anti_d1_sz(tpdm, normalization, bas, opdm.shape[0])
    return np.allclose(opdm_test, opdm)


def check_d2_d1_sz_symm(tpdm, opdm, normalization, bas):
    """
    check the contraction from d2 to d1 for aa and bb matrices
    """
    return np.allclose(opdm, map_d2symm_d1_sz(tpdm, normalization, bas, opdm.shape[0]))


def unspin_adapt(d2aa, d2bb, d2ab):
    """
    Transform a sz_spin-adapted set of 2-RDMs back to the spin-orbtal 2-RDM

    :param d2aa: alpha-alpha block of the 2-RDM.  Antisymmetric basis functions
                 are assumed for this block. block size is r_{s} * (r_{s} - 1)/2
                 where r_{s} is the number of spatial basis functions
    :param d2bb: beta-beta block of the 2-RDM.  Antisymmetric basis functions
                 are assumed for this block. block size is r_{s} * (r_{s} - 1)/2
                 where r_{s} is the number of spatial basis functions
    :param d2ab: alpha-beta block of the 2-RDM. no symmetry adapting is perfomred
                 on this block.  Map directly back to spin-orbital components.
                 This block should have linear dimension r_{s}^{2} where r_{S}
                 is the number of spatial basis functions.
    :return: four-tensor representing the spin-orbital density matrix.
    """
    sp_dim = int(np.sqrt(d2ab.shape[0]))
    so_dim = 2 * sp_dim
    tpdm = np.zeros((so_dim, so_dim, so_dim, so_dim), dtype=complex)

    # build basis look up table
    bas_aa = {}
    bas_ab = {}
    cnt_aa = 0
    cnt_ab = 0
    for p, q in product(range(sp_dim), repeat=2):
        if q > p:
            bas_aa[(p, q)] = cnt_aa
            cnt_aa += 1
        bas_ab[(p, q)] = cnt_ab
        cnt_ab += 1

    # map the d2aa and d2bb back to the spin-orbital 2-RDM
    for p, q, r, s in product(range(sp_dim), repeat=4):
        if p < q and r < s:
            tpdm[2 * p, 2 * q, 2 * r, 2 * s] =  0.5 * d2aa[bas_aa[(p, q)], bas_aa[(r, s)]]
            tpdm[2 * p, 2 * q, 2 * s, 2 * r] = -0.5 * d2aa[bas_aa[(p, q)], bas_aa[(r, s)]]
            tpdm[2 * q, 2 * p, 2 * r, 2 * s] = -0.5 * d2aa[bas_aa[(p, q)], bas_aa[(r, s)]]
            tpdm[2 * q, 2 * p, 2 * s, 2 * r] =  0.5 * d2aa[bas_aa[(p, q)], bas_aa[(r, s)]]

            tpdm[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] =  0.5 * d2bb[bas_aa[(p, q)], bas_aa[(r, s)]]
            tpdm[2 * p + 1, 2 * q + 1, 2 * s + 1, 2 * r + 1] = -0.5 * d2bb[bas_aa[(p, q)], bas_aa[(r, s)]]
            tpdm[2 * q + 1, 2 * p + 1, 2 * r + 1, 2 * s + 1] = -0.5 * d2bb[bas_aa[(p, q)], bas_aa[(r, s)]]
            tpdm[2 * q + 1, 2 * p + 1, 2 * s + 1, 2 * r + 1] =  0.5 * d2bb[bas_aa[(p, q)], bas_aa[(r, s)]]

        tpdm[2 * p, 2 * q + 1, 2 * r, 2 * s + 1] = d2ab[bas_ab[(p, q)], bas_ab[(r, s)]]
        tpdm[2 * q + 1, 2 * p, 2 * r, 2 * s + 1] = -1 * d2ab[bas_ab[(p, q)], bas_ab[(r, s)]]
        tpdm[2 * p, 2 * q + 1, 2 * s + 1, 2 * r] = -1 * d2ab[bas_ab[(p, q)], bas_ab[(r, s)]]
        tpdm[2 * q + 1, 2 * p, 2 * s + 1, 2 * r] = d2ab[bas_ab[(p, q)], bas_ab[(r, s)]]

    return tpdm







