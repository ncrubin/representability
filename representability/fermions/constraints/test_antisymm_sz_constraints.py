import sys
import os
import numpy as np
from itertools import product
from representability.fermions.constraints.antisymm_sz_constraints import (
    trace_d2_aa, trace_d2_bb, trace_d2_ab, d2ab_d1a_mapping,
    d1a_q1a_mapping, d1b_q1b_mapping, d2_q2_mapping, d2_g2_mapping,
    sz_adapted_linear_constraints, d2ab_d1b_mapping, d2aa_d1a_mapping,
    d2bb_d1b_mapping, d2_e2_mapping, s_representability_d2ab,
    sz_representability)
from representability.config import DATA_DIRECTORY
from representability.dualbasis import DualBasis
from representability.fermions.utils import get_molecule_openfermion
from representability.fermions.density.antisymm_sz_density import AntiSymmOrbitalDensity

from representability.tensor import Tensor
from representability.multitensor import MultiTensor

from representability.fermions.basis_utils import geminal_spin_basis
from representability.sampling import add_gaussian_noise
from representability.fermions.density.antisymm_sz_maps import get_sz_spin_adapted
from representability.fermions.hamiltonian import spin_orbital_marginal_norm_min

from openfermion.hamiltonians import MolecularData
from openfermion.transforms import jordan_wigner
from openfermionpsi4 import run_psi4
import openfermion as of


# probably want to upgrade this with yield fixture.  This will need to be an object
def system():
    print('Running System Setup')
    basis = 'sto-3g'
    multiplicity = 1
    charge = 1
    geometry = [('He', [0.0, 0.0, 0.0]), ('H', [0, 0, 0.740848149])]
    molecule = MolecularData(geometry, basis, multiplicity, charge)
    # Run Psi4.
    molecule = run_psi4(molecule,
                        run_scf=True,
                        run_mp2=False,
                        run_cisd=False,
                        run_ccsd=False,
                        run_fci=True,
                        delete_input=False)

    op_mat = of.get_sparse_operator(molecule.get_molecular_hamiltonian()).toarray()
    w, v = np.linalg.eigh(op_mat)
    n_density = v[:, [2]] @ v[:, [2]].conj().T
    rdm_generator = AntiSymmOrbitalDensity(n_density, molecule.n_qubits)
    transform = jordan_wigner
    return n_density, rdm_generator, transform, molecule


def system_hubbard():
    print('Running System Setup')
    import openfermion as of
    U = 1
    sites = 4
    hubbard = of.hamiltonians.fermi_hubbard(1, sites, tunneling=1, coulomb=U,
                                            chemical_potential=0,
                                            magnetic_field=0,
                                            periodic=False,
                                            spinless=False)

    # hamiltonian = of.get_interaction_operator(hubbard)
    op_mat = of.get_number_preserving_sparse_operator(hubbard, 8, 4, spin_preserving=False).toarray()
    w, v = np.linalg.eigh(op_mat)
    gs_e = w[0]
    print(gs_e)
    op_mat = of.get_sparse_operator(hubbard).toarray()
    w, v = np.linalg.eigh(op_mat)
    print(w[0])
    n_density = v[:, [0]] @ v[:, [0]].conj().T
    rdm_generator = AntiSymmOrbitalDensity(n_density, sites * 2)
    return n_density, rdm_generator


def system_h4():
    print('Running System Setup H4')
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    r = 0.75
    geometry = [('H', [0.0, 0.0, 0.0]), ('H', [0, 0, r]), ('H', [0, 0, 2 * r]),
                ('H', [0, 0, 3 * r])]
    molecule = MolecularData(geometry, basis, multiplicity, charge)
    # Run Psi4.
    molecule = run_psi4(molecule,
                        run_scf=True,
                        run_mp2=False,
                        run_cisd=False,
                        run_ccsd=False,
                        run_fci=True,
                        delete_input=False)

    op_mat = of.get_sparse_operator(molecule.get_molecular_hamiltonian()).toarray()
    w, v = np.linalg.eigh(op_mat)
    n_density = v[:, [0]] @ v[:, [0]].conj().T
    rdm_generator = AntiSymmOrbitalDensity(n_density, molecule.n_qubits)
    transform = jordan_wigner
    return n_density, rdm_generator, transform, molecule


def test_d2_trace():
    n_density, rdm_generator, transform, molecule = system()
    assert np.isclose(molecule.fci_energy, -2.84383506834)

    density = AntiSymmOrbitalDensity(n_density, molecule.n_qubits)
    dim = molecule.n_orbitals
    tpdm_aa, tpdm_bb, tpdm_ab, _ = density.construct_tpdm()

    Na, Nb = 1, 1
    trace_ab = 0
    for i, j in product(range(molecule.n_orbitals), repeat=2):
        trace_ab += tpdm_ab[i * dim + j, i * dim + j]
    assert np.isclose(trace_ab, Na, Nb)
    bas_aa, bas_ab = geminal_spin_basis(molecule.n_orbitals)

    tpdm_aa = Tensor(tpdm_aa, name='cckk_aa', basis=bas_aa)
    tpdm_bb = Tensor(tpdm_bb, name='cckk_bb', basis=bas_aa)
    tpdm_ab = Tensor(tpdm_ab, name='cckk_ab', basis=bas_ab)
    rdms = MultiTensor([tpdm_aa, tpdm_bb, tpdm_ab])

    dual_basis = trace_d2_aa(molecule.n_orbitals, molecule.n_electrons / 2)
    rdms.dual_basis = DualBasis(elements=[dual_basis])
    A, b, c = rdms.synthesize_dual_basis()
    Amat =  A.todense()
    bmat = b.todense()
    cmat = c.todense()

    primal_vec = rdms.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))

    dual_basis = trace_d2_bb(molecule.n_orbitals, molecule.n_electrons / 2)
    rdms.dual_basis = DualBasis(elements=[dual_basis])
    A, _, c = rdms.synthesize_dual_basis()
    Amat =  A.todense()
    cmat = c.todense()

    primal_vec = rdms.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))

    dual_basis = trace_d2_ab(molecule.n_orbitals, molecule.n_electrons / 2, molecule.n_electrons / 2)
    rdms.dual_basis = DualBasis(elements=[dual_basis])
    A, _, c = rdms.synthesize_dual_basis()
    Amat =  A.todense()
    cmat = c.todense()

    primal_vec = rdms.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))

    db = DualBasis()
    db += trace_d2_aa(molecule.n_orbitals, molecule.n_electrons / 2)
    db += trace_d2_ab(molecule.n_orbitals, molecule.n_electrons / 2, molecule.n_electrons / 2)
    db += trace_d2_bb(molecule.n_orbitals, molecule.n_electrons / 2)
    rdms.dual_basis = db
    A, _, c = rdms.synthesize_dual_basis()
    Amat =  A.todense()
    cmat = c.todense()

    primal_vec = rdms.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))


def test_d2_trace_h4():
    n_density, rdm_generator, transform, molecule = system_h4()

    density = AntiSymmOrbitalDensity(n_density, molecule.n_qubits)
    dim = molecule.n_orbitals
    tpdm_aa, tpdm_bb, tpdm_ab, _ = density.construct_tpdm()

    Na, Nb = 2, 2
    trace_ab = 0
    for i, j in product(range(molecule.n_orbitals), repeat=2):
        trace_ab += tpdm_ab[i * dim + j, i * dim + j]
    assert np.isclose(trace_ab, Na, Nb)
    assert np.isclose(np.trace(tpdm_aa), Na * (Na - 1) / 2)
    assert np.isclose(np.trace(tpdm_bb), Nb * (Nb - 1) / 2)
    bas_aa, bas_ab = geminal_spin_basis(molecule.n_orbitals)

    tpdm_aa = Tensor(tpdm_aa, name='cckk_aa', basis=bas_aa)
    tpdm_bb = Tensor(tpdm_bb, name='cckk_bb', basis=bas_aa)
    tpdm_ab = Tensor(tpdm_ab, name='cckk_ab', basis=bas_ab)
    rdms = MultiTensor([tpdm_aa, tpdm_bb, tpdm_ab])

    dual_basis = trace_d2_aa(molecule.n_orbitals, molecule.n_electrons / 2)
    rdms.dual_basis = DualBasis(elements=[dual_basis])
    A, b, c = rdms.synthesize_dual_basis()
    Amat = A.todense()
    bmat = b.todense()
    cmat = c.todense()

    primal_vec = rdms.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))

    dual_basis = trace_d2_bb(molecule.n_orbitals, molecule.n_electrons / 2)
    rdms.dual_basis = DualBasis(elements=[dual_basis])
    A, _, c = rdms.synthesize_dual_basis()
    Amat = A.todense()
    cmat = c.todense()

    primal_vec = rdms.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))

    dual_basis = trace_d2_ab(molecule.n_orbitals, molecule.n_electrons / 2,
                             molecule.n_electrons / 2)
    rdms.dual_basis = DualBasis(elements=[dual_basis])
    A, _, c = rdms.synthesize_dual_basis()
    Amat = A.todense()
    cmat = c.todense()

    primal_vec = rdms.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))

    db = DualBasis()
    db += trace_d2_aa(molecule.n_orbitals, molecule.n_electrons / 2)
    db += trace_d2_ab(molecule.n_orbitals, molecule.n_electrons / 2,
                      molecule.n_electrons / 2)
    db += trace_d2_bb(molecule.n_orbitals, molecule.n_electrons / 2)
    rdms.dual_basis = db
    A, _, c = rdms.synthesize_dual_basis()
    Amat = A.todense()
    cmat = c.todense()

    primal_vec = rdms.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))


def test_d2_trace_hubbard():
    n_density, rdm_generator = system_hubbard()

    density = AntiSymmOrbitalDensity(n_density, 8)
    tpdm_aa, tpdm_bb, tpdm_ab, _ = density.construct_tpdm()
    bas_aa, bas_ab = geminal_spin_basis(4)

    tpdm_aa = Tensor(tpdm_aa, name='cckk_aa', basis=bas_aa)
    tpdm_bb = Tensor(tpdm_bb, name='cckk_bb', basis=bas_aa)
    tpdm_ab = Tensor(tpdm_ab, name='cckk_ab', basis=bas_ab)
    rdms = MultiTensor([tpdm_aa, tpdm_bb, tpdm_ab])

    dual_basis = trace_d2_aa(4, 2)
    rdms.dual_basis = DualBasis(elements=[dual_basis])
    A, b, c = rdms.synthesize_dual_basis()
    Amat =  A.todense()
    bmat = b.todense()
    cmat = c.todense()

    primal_vec = rdms.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))

    dual_basis = trace_d2_bb(4, 2)
    rdms.dual_basis = DualBasis(elements=[dual_basis])
    A, _, c = rdms.synthesize_dual_basis()
    Amat =  A.todense()
    cmat = c.todense()

    primal_vec = rdms.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))

    dual_basis = trace_d2_ab(4, 2, 2)
    rdms.dual_basis = DualBasis(elements=[dual_basis])
    A, _, c = rdms.synthesize_dual_basis()
    Amat =  A.todense()
    cmat = c.todense()

    primal_vec = rdms.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))

    db = DualBasis()
    db += trace_d2_aa(4, 2)
    db += trace_d2_ab(4, 2, 2)
    db += trace_d2_bb(4, 2)
    rdms.dual_basis = db
    A, _, c = rdms.synthesize_dual_basis()
    Amat =  A.todense()
    cmat = c.todense()

    primal_vec = rdms.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))


def test_d2_spin_rep():
    n_density, rdm_generator, transform, molecule = system()
    assert np.isclose(molecule.fci_energy, -2.84383506834)

    density = AntiSymmOrbitalDensity(n_density, molecule.n_qubits)
    tpdm_aa, tpdm_bb, tpdm_ab, _ = density.construct_tpdm()
    bas_aa, bas_ab = geminal_spin_basis(molecule.n_orbitals)

    tpdm_ab = Tensor(tpdm_ab, name='cckk_ab', basis=bas_ab)
    rdms = MultiTensor([tpdm_ab])

    dim = int(np.sqrt(tpdm_ab.data.shape[0]))
    s_rep_dual_constant = 0
    for i, j in product(range(dim), repeat=2):
        s_rep_dual_constant += tpdm_ab.data[bas_ab.rev((i, j)), bas_ab.rev((j, i))]

    N = molecule.n_electrons
    M = 0
    S = 0
    db = DualBasis()
    db += s_representability_d2ab(dim, N, M, S)
    rdms.dual_basis = db
    xvec = rdms.vectorize_tensors()
    A, _, b = rdms.synthesize_dual_basis()
    assert np.allclose(A.dot(xvec) - b, 0.0)
    assert np.allclose(A.dot(xvec), s_rep_dual_constant)


def test_d2_spin_rep_h4():
    n_density, rdm_generator, transform, molecule = system_h4()

    density = AntiSymmOrbitalDensity(n_density, molecule.n_qubits)
    tpdm_aa, tpdm_bb, tpdm_ab, _ = density.construct_tpdm()
    bas_aa, bas_ab = geminal_spin_basis(molecule.n_orbitals)

    tpdm_ab = Tensor(tpdm_ab, name='cckk_ab', basis=bas_ab)
    rdms = MultiTensor([tpdm_ab])

    dim = int(np.sqrt(tpdm_ab.data.shape[0]))
    s_rep_dual_constant = 0
    for i, j in product(range(dim), repeat=2):
        s_rep_dual_constant += tpdm_ab.data[bas_ab.rev((i, j)), bas_ab.rev((j, i))]

    N = molecule.n_electrons
    M = 0
    S = 0
    db = DualBasis()
    db += s_representability_d2ab(dim, N, M, S)
    rdms.dual_basis = db
    xvec = rdms.vectorize_tensors()
    A, _, b = rdms.synthesize_dual_basis()
    assert np.allclose(A.dot(xvec) - b, 0.0)
    assert np.allclose(A.dot(xvec), s_rep_dual_constant)


def test_d2_spin_rep_hubbard():
    n_density, rdm_generator = system_hubbard()

    density = AntiSymmOrbitalDensity(n_density, 8)
    tpdm_aa, tpdm_bb, tpdm_ab, _ = density.construct_tpdm()
    bas_aa, bas_ab = geminal_spin_basis(4)

    tpdm_ab = Tensor(tpdm_ab, name='cckk_ab', basis=bas_ab)
    rdms = MultiTensor([tpdm_ab])

    dim = int(np.sqrt(tpdm_ab.data.shape[0]))
    s_rep_dual_constant = 0
    for i, j in product(range(dim), repeat=2):
        s_rep_dual_constant += tpdm_ab.data[bas_ab.rev((i, j)), bas_ab.rev((j, i))]

    N = 4
    M = 0
    S = 0
    db = DualBasis()
    db += s_representability_d2ab(dim, N, M, S)
    rdms.dual_basis = db
    xvec = rdms.vectorize_tensors()
    A, _, b = rdms.synthesize_dual_basis()
    assert np.allclose(A.dot(xvec) - b, 0.0)
    assert np.allclose(A.dot(xvec), s_rep_dual_constant)


def test_d2_spin_sz_rep():
    n_density, rdm_generator, transform, molecule = system()
    assert np.isclose(molecule.fci_energy, -2.84383506834)

    density = AntiSymmOrbitalDensity(n_density, molecule.n_qubits)
    tpdm_aa, tpdm_bb, tpdm_ab, _ = density.construct_tpdm()
    opdm_a, opdm_b = density.construct_opdm()
    bas_aa, bas_ab = geminal_spin_basis(molecule.n_orbitals)

    tpdm_ab = Tensor(tpdm_ab, name='cckk_ab', basis=bas_ab)
    opdm_a = Tensor(opdm_a, name='ck_a')
    opdm_b = Tensor(opdm_b, name='ck_b')
    rdms = MultiTensor([opdm_a, opdm_b, tpdm_ab])

    dim = int(np.sqrt(tpdm_ab.data.shape[0]))
    sz_rep_value = 0
    for i in range(dim):
        sz_rep_value += 0.5 * (opdm_a.data[i, i] - opdm_b.data[i, i])

    N = molecule.n_electrons
    M = 0
    S = 0
    db = DualBasis()
    db += s_representability_d2ab(dim, N, M, S)
    db += sz_representability(dim, M)
    rdms.dual_basis = db
    xvec = rdms.vectorize_tensors()
    A, _, b = rdms.synthesize_dual_basis()
    assert np.allclose(A.dot(xvec) - b, 0.0)


def test_d2_d1_mapping():
    n_density, rdm_generator, transform, molecule = system_h4()

    density = AntiSymmOrbitalDensity(n_density, molecule.n_qubits)
    tpdm_aa, tpdm_bb, tpdm_ab, [bas_aa, bas_ab] = density.construct_tpdm()
    opdm_a, opdm_b = density.construct_opdm()

    from itertools import product
    test_opdm = np.zeros_like(opdm_a)
    for i, j in product(range(opdm_a.shape[0]), repeat=2):
        if i <= j:
            for r in range(opdm_a.shape[0]):
                if i != r and j != r:
                    top_gem = tuple(sorted([i, r]))
                    bot_gem = tuple(sorted([j, r]))
                    parity = (-1)**(r < i) * (-1)**(r < j)
                    if i == j:
                        test_opdm[i, j] += tpdm_aa[bas_aa[top_gem], bas_aa[bot_gem]] * parity
                    else:
                        test_opdm[j, i] += tpdm_aa[bas_aa[top_gem], bas_aa[bot_gem]] * parity * 0.5
                        test_opdm[j, i] += tpdm_aa[bas_aa[bot_gem], bas_aa[top_gem]] * parity * 0.5
                        test_opdm[i, j] += tpdm_aa[bas_aa[top_gem], bas_aa[bot_gem]] * parity * 0.5
                        test_opdm[i, j] += tpdm_aa[bas_aa[bot_gem], bas_aa[top_gem]] * parity * 0.5

    assert np.allclose(test_opdm, opdm_a)

    bas_aa, bas_ab = geminal_spin_basis(molecule.n_orbitals)
    opdm_a = Tensor(opdm_a, name='ck_a')
    opdm_b = Tensor(opdm_b, name='ck_b')
    tpdm_aa = Tensor(tpdm_aa, name='cckk_aa', basis=bas_aa)
    tpdm_bb = Tensor(tpdm_bb, name='cckk_bb', basis=bas_aa)
    tpdm_ab = Tensor(tpdm_ab, name='cckk_ab', basis=bas_ab)
    rdms = MultiTensor([opdm_a, opdm_b, tpdm_aa, tpdm_bb, tpdm_ab])

    # d2ab_d1a test
    dual_basis = DualBasis()
    dual_basis = d2ab_d1a_mapping(molecule.n_orbitals, molecule.n_electrons / 2)
    dual_basis += d2ab_d1b_mapping(molecule.n_orbitals, molecule.n_electrons / 2)
    dual_basis += d2aa_d1a_mapping(molecule.n_orbitals, molecule.n_electrons / 2)
    dual_basis += d2bb_d1b_mapping(molecule.n_orbitals, molecule.n_electrons / 2)
    rdms.dual_basis = dual_basis
    A, _, c = rdms.synthesize_dual_basis()
    Amat =  A.todense()
    w, v = np.linalg.eigh(np.dot(Amat, Amat.T))
    cmat = c.todense()

    primal_vec = rdms.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))


def test_d2_d1_mapping_hubbard():
    n_density, rdm_generator = system_hubbard()

    density = AntiSymmOrbitalDensity(n_density, 8)
    tpdm_aa, tpdm_bb, tpdm_ab, [bas_aa, bas_ab] = density.construct_tpdm()
    opdm_a, opdm_b = density.construct_opdm()


    from itertools import product
    test_opdm = np.zeros_like(opdm_a)
    for i, j in product(range(opdm_a.shape[0]), repeat=2):
        if i <= j:
            for r in range(opdm_a.shape[0]):
                if i != r and j != r:
                    top_gem = tuple(sorted([i, r]))
                    bot_gem = tuple(sorted([j, r]))
                    parity = (-1)**(r < i) * (-1)**(r < j)
                    if i == j:
                        test_opdm[i, j] += tpdm_aa[bas_aa[top_gem], bas_aa[bot_gem]] * parity
                    else:
                        test_opdm[j, i] += tpdm_aa[bas_aa[top_gem], bas_aa[bot_gem]] * parity * 0.5
                        test_opdm[j, i] += tpdm_aa[bas_aa[bot_gem], bas_aa[top_gem]] * parity * 0.5
                        test_opdm[i, j] += tpdm_aa[bas_aa[top_gem], bas_aa[bot_gem]] * parity * 0.5
                        test_opdm[i, j] += tpdm_aa[bas_aa[bot_gem], bas_aa[top_gem]] * parity * 0.5

    assert np.allclose(test_opdm, opdm_a)

    bas_aa, bas_ab = geminal_spin_basis(4)
    opdm_a = Tensor(opdm_a, name='ck_a')
    opdm_b = Tensor(opdm_b, name='ck_b')
    tpdm_aa = Tensor(tpdm_aa, name='cckk_aa', basis=bas_aa)
    tpdm_bb = Tensor(tpdm_bb, name='cckk_bb', basis=bas_aa)
    tpdm_ab = Tensor(tpdm_ab, name='cckk_ab', basis=bas_ab)
    rdms = MultiTensor([opdm_a, opdm_b, tpdm_aa, tpdm_bb, tpdm_ab])

    # d2ab_d1a test
    Na = Nb = 2
    dual_basis =  d2ab_d1a_mapping(4, Nb)
    dual_basis += d2ab_d1b_mapping(4, Na)
    dual_basis += d2aa_d1a_mapping(4, Na)
    dual_basis += d2bb_d1b_mapping(4, Nb)
    rdms.dual_basis = dual_basis
    A, _, c = rdms.synthesize_dual_basis()
    Amat =  A.todense()
    w, v = np.linalg.eigh(np.dot(Amat, Amat.T))
    cmat = c.todense()

    primal_vec = rdms.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))


def test_d1_q1_mapping():
    n_density, rdm_generator, transform, molecule = system_h4()

    density = AntiSymmOrbitalDensity(n_density, molecule.n_qubits)
    tpdm_aa, tpdm_bb, tpdm_ab, _ = density.construct_tpdm()
    opdm_a, opdm_b = density.construct_opdm()
    oqdm_a, oqdm_b = density.construct_ohdm()

    bas_aa, bas_ab = geminal_spin_basis(molecule.n_orbitals)

    opdm_a = Tensor(opdm_a, name='ck_a')
    opdm_b = Tensor(opdm_b, name='ck_b')
    oqdm_a = Tensor(oqdm_a, name='kc_a')
    oqdm_b = Tensor(oqdm_b, name='kc_b')
    rdms = MultiTensor([opdm_a, opdm_b, oqdm_a, oqdm_b])

    dual_basis = d1a_q1a_mapping(molecule.n_orbitals)
    rdms.dual_basis = dual_basis
    A, _, c = rdms.synthesize_dual_basis()
    Amat =  A.todense()
    cmat = c.todense()

    primal_vec = rdms.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))

    dual_basis = d1b_q1b_mapping(molecule.n_orbitals)
    rdms.dual_basis = dual_basis
    A, _, c = rdms.synthesize_dual_basis()
    Amat =  A.todense()
    cmat = c.todense()

    primal_vec = rdms.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))

    dual_basis_a = d1b_q1b_mapping(molecule.n_orbitals)
    dual_basis_b = d1a_q1a_mapping(molecule.n_orbitals)
    rdms.dual_basis = dual_basis_a + dual_basis_b
    A, _, c = rdms.synthesize_dual_basis()
    Amat =  A.todense()
    cmat = c.todense()

    primal_vec = rdms.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))


def test_d2_q2_mapping():
    n_density, rdm_generator, transform, molecule = system_h4()

    density = AntiSymmOrbitalDensity(n_density, molecule.n_qubits)
    tpdm_aa, tpdm_bb, tpdm_ab, _ = density.construct_tpdm()
    tqdm_aa, tqdm_bb, tqdm_ab, _ = density.construct_thdm()
    opdm_a, opdm_b = density.construct_opdm()
    bas_aa, bas_ab = geminal_spin_basis(molecule.n_orbitals)

    opdm_a = Tensor(opdm_a, name='ck_a')
    opdm_b = Tensor(opdm_b, name='ck_b')
    tpdm_aa = Tensor(tpdm_aa, name='cckk_aa', basis=bas_aa)
    tpdm_bb = Tensor(tpdm_bb, name='cckk_bb', basis=bas_aa)
    tpdm_ab = Tensor(tpdm_ab, name='cckk_ab', basis=bas_ab)
    tqdm_aa = Tensor(tqdm_aa, name='kkcc_aa', basis=bas_aa)
    tqdm_bb = Tensor(tqdm_bb, name='kkcc_bb', basis=bas_aa)
    tqdm_ab = Tensor(tqdm_ab, name='kkcc_ab', basis=bas_ab)

    rdms = MultiTensor([opdm_a, opdm_b, tpdm_aa, tpdm_bb, tpdm_ab, tqdm_aa, tqdm_bb, tqdm_ab])
    dual_basis = d2_q2_mapping(molecule.n_orbitals)
    rdms.dual_basis = dual_basis
    A, _, c = rdms.synthesize_dual_basis()
    Amat =  A.todense()
    cmat = c.todense()

    primal_vec = rdms.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))


def test_d2_q2_mapping_hubbard():
    n_density, rdm_generator = system_hubbard()
    density = AntiSymmOrbitalDensity(n_density, 8)
    tpdm_aa, tpdm_bb, tpdm_ab, _ = density.construct_tpdm()
    tqdm_aa, tqdm_bb, tqdm_ab, _ = density.construct_thdm()
    opdm_a, opdm_b = density.construct_opdm()
    bas_aa, bas_ab = geminal_spin_basis(4)

    opdm_a = Tensor(opdm_a, name='ck_a')
    opdm_b = Tensor(opdm_b, name='ck_b')
    tpdm_aa = Tensor(tpdm_aa, name='cckk_aa', basis=bas_aa)
    tpdm_bb = Tensor(tpdm_bb, name='cckk_bb', basis=bas_aa)
    tpdm_ab = Tensor(tpdm_ab, name='cckk_ab', basis=bas_ab)
    tqdm_aa = Tensor(tqdm_aa, name='kkcc_aa', basis=bas_aa)
    tqdm_bb = Tensor(tqdm_bb, name='kkcc_bb', basis=bas_aa)
    tqdm_ab = Tensor(tqdm_ab, name='kkcc_ab', basis=bas_ab)

    rdms = MultiTensor([opdm_a, opdm_b, tpdm_aa, tpdm_bb, tpdm_ab, tqdm_aa, tqdm_bb, tqdm_ab])
    dual_basis = d2_q2_mapping(4)
    rdms.dual_basis = dual_basis
    A, _, c = rdms.synthesize_dual_basis()
    Amat =  A.todense()
    cmat = c.todense()

    primal_vec = rdms.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))


def test_d2_g2_mapping():
    n_density, rdm_generator, transform, molecule = system_h4()
    # n_density, rdm_generator = system_hubbard()
    # n_density, rdm_generator, transform, molecule = system()
    dim = 4
    density = AntiSymmOrbitalDensity(n_density, 2 * dim)
    tpdm_aa, tpdm_bb, tpdm_ab, _ = density.construct_tpdm()
    tqdm_aa, tqdm_bb, tqdm_ab, _ = density.construct_thdm()
    phdm_ab, phdm_ba, phdm_aabb = density.construct_phdm()
    # for i, j, k, l in product(range(dim), repeat=4):
    #     fop = ((2 * i, 1), (2 * j + 1, 0), (2 * l + 1, 1), (2 * k, 0))
    #     fop = of.FermionOperator(fop)
    #     opmat = of.get_sparse_operator(fop, n_qubits=2 * dim)
    #     rdm_val = (np.trace(n_density @ opmat))
    #     # print(rdm_val, phdm_ab[i * dim + j, k * dim + l])
    #     assert np.isclose(rdm_val, phdm_ab[i * dim + j, k * dim + l])

    # for i, j, k, l in product(range(dim), repeat=4):
    #     fop = ((2 * i + 1, 1), (2 * j, 0), (2 * l, 1), (2 * k + 1, 0))
    #     fop = of.FermionOperator(fop)
    #     opmat = of.get_sparse_operator(fop, n_qubits=2 * dim)
    #     rdm_val = (np.trace(n_density @ opmat))
    #     assert np.isclose(rdm_val, phdm_ba[i * dim + j, k * dim + l])

    # for i, j, k, l in product(range(dim), repeat=4):
    #     fop = ((2 * i, 1), (2 * j, 0), (2 * k, 1), (2 * l, 0))
    #     fop = of.FermionOperator(fop)
    #     opmat = of.get_sparse_operator(of.jordan_wigner(fop), n_qubits=2 * dim)
    #     rdm_val = (np.trace(n_density @ opmat))
    #     assert np.isclose(rdm_val, phdm_aabb[i * dim + j, l * dim + k])

    # for i, j, k, l in product(range(dim), repeat=4):
    #     fop = ((2 * i + 1, 1), (2 * j + 1, 0), (2 * k + 1, 1), (2 * l + 1, 0))
    #     fop = of.FermionOperator(fop)
    #     opmat = of.get_sparse_operator(of.jordan_wigner(fop), n_qubits=2 * dim)
    #     rdm_val = (np.trace(n_density @ opmat))
    #     # print((i, j, k, l), rdm_val, phdm_aabb[i * dim + j + dim**2, l * dim + k + dim**2])
    #     assert np.isclose(rdm_val, phdm_aabb[i * dim + j + dim**2, l * dim + k + dim**2])

    # for i, j, k, l in product(range(dim), repeat=4):
    #     fop = ((2 * i, 1), (2 * j, 0), (2 * k + 1, 1), (2 * l + 1, 0))
    #     fop = of.FermionOperator(fop)
    #     opmat = of.get_sparse_operator(of.jordan_wigner(fop), n_qubits=2 * dim)
    #     rdm_val = (np.trace(n_density @ opmat))
    #     assert np.isclose(rdm_val, phdm_aabb[i * dim + j, l * dim + k + dim**2])

    # for i, j, k, l in product(range(dim), repeat=4):
    #     fop = ((2 * i + 1, 1), (2 * j + 1, 0), (2 * l, 1), (2 * k, 0))
    #     fop = of.FermionOperator(fop)
    #     opmat = of.get_sparse_operator(of.jordan_wigner(fop), n_qubits=2 * dim)
    #     rdm_val = (np.trace(n_density @ opmat))
    #     # print((i, j, k, l), rdm_val, phdm_aabb[i * dim + j + dim**2, k * dim + l])
    #     assert np.isclose(rdm_val, phdm_aabb[i * dim + j + dim**2, k * dim + l])

    # for i, j, k, l in product(range(dim), repeat=4):
    #     assert np.isclose(phdm_aabb[i * dim + j + dim**2, k * dim + l],
    #                       phdm_aabb[k * dim + l, i * dim + j + dim**2]
    #                       )

    assert of.is_hermitian(phdm_ab)
    assert of.is_hermitian(phdm_ba)
    assert of.is_hermitian(phdm_aabb)
    w, v = np.linalg.eigh(phdm_ab)
    assert np.all(w > -1.0E-14)
    w, v = np.linalg.eigh(phdm_ba)
    assert np.all(w > -1.0E-14)
    w, v = np.linalg.eigh(phdm_aabb)
    assert np.all(w > -1.0E-14)

    opdm_a, opdm_b = density.construct_opdm()
    bas_aa, bas_ab = geminal_spin_basis(dim)

    opdm_a = Tensor(opdm_a, name='ck_a')
    opdm_b = Tensor(opdm_b, name='ck_b')
    tpdm_aa = Tensor(tpdm_aa, name='cckk_aa', basis=bas_aa)
    tpdm_bb = Tensor(tpdm_bb, name='cckk_bb', basis=bas_aa)
    tpdm_ab = Tensor(tpdm_ab, name='cckk_ab', basis=bas_ab)
    tqdm_aa = Tensor(tqdm_aa, name='kkcc_aa', basis=bas_aa)
    tqdm_bb = Tensor(tqdm_bb, name='kkcc_bb', basis=bas_aa)
    tqdm_ab = Tensor(tqdm_ab, name='kkcc_ab', basis=bas_ab)
    phdm_ab = Tensor(phdm_ab, name='ckck_ab', basis=bas_ab)
    phdm_ba = Tensor(phdm_ba, name='ckck_ba', basis=bas_ab)
    phdm_aabb = Tensor(phdm_aabb, name='ckck_aabb')  # What basis do we want to use for super blocks like this?

    rdms = MultiTensor([opdm_a, opdm_b, tpdm_aa, tpdm_bb, tpdm_ab, tqdm_aa, tqdm_bb, tqdm_ab,
                        phdm_ab, phdm_ba, phdm_aabb])
    dual_basis = d2_g2_mapping(dim)
    rdms.dual_basis = dual_basis

    A, _, c = rdms.synthesize_dual_basis()
    Amat = A.todense()
    cmat = c.todense()

    primal_vec = rdms.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))


def test_d2_g2_mapping_hubbard():
    n_density, rdm_generator = system_hubbard()

    density = AntiSymmOrbitalDensity(n_density, 8)
    tpdm_aa, tpdm_bb, tpdm_ab, _ = density.construct_tpdm()
    tqdm_aa, tqdm_bb, tqdm_ab, _ = density.construct_thdm()
    phdm_ab, phdm_ba, phdm_aabb = density.construct_phdm()
    opdm_a, opdm_b = density.construct_opdm()
    bas_aa, bas_ab = geminal_spin_basis(4)

    opdm_a = Tensor(opdm_a, name='ck_a')
    opdm_b = Tensor(opdm_b, name='ck_b')
    tpdm_aa = Tensor(tpdm_aa, name='cckk_aa', basis=bas_aa)
    tpdm_bb = Tensor(tpdm_bb, name='cckk_bb', basis=bas_aa)
    tpdm_ab = Tensor(tpdm_ab, name='cckk_ab', basis=bas_ab)
    tqdm_aa = Tensor(tqdm_aa, name='kkcc_aa', basis=bas_aa)
    tqdm_bb = Tensor(tqdm_bb, name='kkcc_bb', basis=bas_aa)
    tqdm_ab = Tensor(tqdm_ab, name='kkcc_ab', basis=bas_ab)
    phdm_ab = Tensor(phdm_ab, name='ckck_ab', basis=bas_ab)
    phdm_ba = Tensor(phdm_ba, name='ckck_ba', basis=bas_ab)
    phdm_aabb = Tensor(phdm_aabb, name='ckck_aabb')  # What basis do we want to use for super blocks like this?

    rdms = MultiTensor([opdm_a, opdm_b, tpdm_aa, tpdm_bb, tpdm_ab, tqdm_aa, tqdm_bb, tqdm_ab,
                        phdm_ab, phdm_ba, phdm_aabb])
    dual_basis = d2_g2_mapping(4)
    rdms.dual_basis = dual_basis
    A, _, c = rdms.synthesize_dual_basis()
    Amat = A.todense()
    cmat = c.todense()

    primal_vec = rdms.vectorize_tensors()
    residual = Amat.dot(primal_vec) - cmat
    assert np.allclose(residual, np.zeros_like(residual))


def test_d2_e2_constraint():
    n_density, rdm_generator, transform, molecule = system()

    density = AntiSymmOrbitalDensity(n_density, molecule.n_qubits)
    tpdm_aa, tpdm_bb, tpdm_ab, [bas_aa, bas_ab] = density.construct_tpdm()
    tqdm_aa, tqdm_bb, tqdm_ab, _ = density.construct_thdm()

    # generate error tensors
    corrupted_tpdm_aa = add_gaussian_noise(tpdm_aa, 1)
    corrupted_tpdm_aa = 0.5 * (corrupted_tpdm_aa + corrupted_tpdm_aa.T)
    corrupted_tpdm_bb = add_gaussian_noise(tpdm_bb, 1)
    corrupted_tpdm_bb = 0.5 * (corrupted_tpdm_bb + corrupted_tpdm_bb.T)
    corrupted_tpdm_ab = add_gaussian_noise(tpdm_ab, 1)
    corrupted_tpdm_ab = 0.5 * (corrupted_tpdm_ab + corrupted_tpdm_ab.T)

    error_aa = corrupted_tpdm_aa - tpdm_aa
    error_bb = corrupted_tpdm_bb - tpdm_bb
    error_ab = corrupted_tpdm_ab - tpdm_ab

    error_matrix_aa = density.construct_tpdm_error_matrix(error_aa)
    error_matrix_bb = density.construct_tpdm_error_matrix(error_bb)
    error_matrix_ab = density.construct_tpdm_error_matrix(error_ab)

    assert error_matrix_aa.shape[0] == 2 * tpdm_aa.shape[0]
    assert error_matrix_bb.shape[0] == 2 * tpdm_bb.shape[0]
    assert error_matrix_ab.shape[0] == 2 * tpdm_ab.shape[0]

    dim_aa = tpdm_aa.shape[0]
    dim_bb = tpdm_bb.shape[0]
    dim_ab = tpdm_ab.shape[0]

    assert np.allclose(error_matrix_aa[:dim_aa, :dim_aa], np.eye(dim_aa))
    assert np.allclose(error_matrix_bb[:dim_bb, :dim_bb], np.eye(dim_bb))
    assert np.allclose(error_matrix_ab[:dim_ab, :dim_ab], np.eye(dim_ab))

    assert np.allclose(error_matrix_aa[:dim_aa, dim_aa:], error_aa)
    assert np.allclose(error_matrix_aa[dim_aa:, :dim_aa], error_aa.T)

    assert np.allclose(error_matrix_bb[:dim_bb, dim_bb:], error_bb)
    assert np.allclose(error_matrix_bb[dim_bb:, :dim_bb], error_bb.T)

    assert np.allclose(error_matrix_ab[:dim_ab, dim_ab:], error_ab)
    assert np.allclose(error_matrix_ab[dim_ab:, :dim_ab], error_ab.T)

    # get basis bijection
    bij_bas_aa, bij_bas_ab = geminal_spin_basis(molecule.n_orbitals)

    tpdm_aa = Tensor(tensor=tpdm_aa, name='cckk_aa', basis=bij_bas_aa)
    error_tensor_aa = Tensor(tensor=error_matrix_aa, name='cckk_me_aa')
    tpdm_bb = Tensor(tensor=tpdm_bb, name='cckk_bb', basis=bij_bas_aa)
    error_tensor_bb = Tensor(tensor=error_matrix_bb, name='cckk_me_bb')
    tpdm_ab = Tensor(tensor=tpdm_ab, name='cckk_ab', basis=bij_bas_ab)
    error_tensor_ab = Tensor(tensor=error_matrix_ab, name='cckk_me_ab')
    rdms = MultiTensor([tpdm_aa, tpdm_bb, tpdm_ab, error_tensor_aa, error_tensor_bb, error_tensor_ab])
    xvec = rdms.vectorize_tensors()

    db = d2_e2_mapping(int(np.sqrt(dim_ab)), bas_aa, bas_ab, corrupted_tpdm_aa, corrupted_tpdm_bb, corrupted_tpdm_ab)
    rdms.dual_basis = db
    A, _, b = rdms.synthesize_dual_basis()
    A = A.todense()
    b = b.todense()

    w, v = np.linalg.eigh(A.dot(A.T))
    assert np.all(w > 0)

    assert np.allclose(A.dot(xvec) - b, 0.0)

if __name__ == "__main__":
    # test_d2_trace()
    # test_d2_trace_h4()
    # test_d2_trace_hubbard()
    # test_d2_spin_rep()
    # test_d2_spin_rep_h4()
    # test_d2_spin_rep_hubbard()
    # test_d2_d1_mapping()
    # test_d2_d1_mapping_hubbard()
    # test_d1_q1_mapping()
    # test_d2_q2_mapping()
    # test_d2_q2_mapping_hubbard()
    test_d2_g2_mapping()
    test_d2_g2_mapping_hubbard()
