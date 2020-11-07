"""
Generates an SDP object with cvxpy and solve
"""
import cvxpy as cvx
import numpy as np
from representability.fermions.hamiltonian import spin_adapted_interaction_tensor_rdm_consistent, make_sz_spin_adapted_hamiltonian
from representability.fermions.hamiltonian import spin_orbital_interaction_tensor
from representability.fermions.constraints.antisymm_sz_constraints import sz_adapted_linear_constraints
from representability.multitensor import MultiTensor
from representability.tensor import Tensor
from representability.fermions.basis_utils import geminal_spin_basis
from representability.fermions.constraints.spin_orbital_constraints import spin_orbital_linear_constraints

from representability.fermions.utils import write_sdpfile
from representability.fermions.density.antisymm_sz_density import unspin_adapt

from sdpsolve.sdp import SDP

from sdpsolve.solvers.bpsdp import solve_bpsdp
from sdpsolve.solvers.rrsdp import solve_rrsdp
# from sdpsolve.solvers.bpsdp.bpsdp_old import solve_bpsdp
from sdpsolve.utils.matreshape import vec2block


import itertools, numpy
from itertools import product
import openfermion as of
import itertools


def gen_trans_2rdm(gem_dim, bas_dim):
    bas = dict(zip(range(gem_dim), itertools.product(range(1, bas_dim + 1),
                                                     range(1, bas_dim + 1))))

    bas_rev = dict(zip(bas.values(), bas.keys()))
    # print "bas_rev", bas_rev

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

    trans_mat = numpy.zeros((gem_dim, gem_dim))
    cnt = 0
    for xx in D2ab_abas.keys():
        i, j = D2ab_abas[xx]
        x1 = bas_rev[(i, j)]
        #     print x1
        x2 = bas_rev[(j, i)]
        #  print x2
        #  print cnt
        #  print "another turn"
        trans_mat[x1, cnt] = 1. / numpy.sqrt(2)
        trans_mat[x2, cnt] = -1. / numpy.sqrt(2)
        cnt += 1
    #  print "trans_mat for D2s begins"
    for xx in D2ab_sbas.keys():
        i, j = D2ab_sbas[xx]
        x1 = bas_rev[(i, j)]
        x2 = bas_rev[(j, i)]
        #  print x1
        #  print x2
        if x1 == x2:
            trans_mat[x1, cnt] = 1.0
        #       print cnt
        else:
            #      print cnt
            trans_mat[x1, cnt] = 1. / numpy.sqrt(2)
            trans_mat[x2, cnt] = 1. / numpy.sqrt(2)
        #  print "another turn"
        cnt += 1
    # print "trans_mat"
    # print trans_mat
    return trans_mat


def get_var_indices(tensor, element):
    """
    generate matrix coordinate given the tensor and element reference

    :param Tensor tensor: Tensor object from representability
    :param element: tuple of elements corresponding to the coordinate in
                    tensor form.  For a matrix tensor element will be (a, b)
                    corresponding to the (row, col) index.  For a four index
                    tensor element will be (a, b, c, d) and the corresponding
                    c-ordered element will be returned
    :return: matrix coordinate
    """
    vector_index = tensor.index_vectorized(*element)
    lin_dim = int(np.sqrt(tensor.size))
    return (vector_index // lin_dim, vector_index % lin_dim)


def v2rdm_cvvxpy(one_body_ints, two_body_ints, Na, Nb):
    """
    Generate a cvxpy Problem corresponding the variational 2-RDM method

    Note: if you are using with integrals generated from OpenFermion
    you need to reorder the 2-body ints before passing to this function.
    this can be accomplished by changing the integrals using einsum
    np.einsum('ijkl->ijlk', two_body_ints). the integrals can be found after
    grabbing the hamiltonian from molecule.get_molecular_hamiltonian() and calling
    the attribute `two_body_tensor`.

    This routine returns a cvx problem corresponding to the v2-RDM DQG sdp.
    This can be solved by calling cvx.Problem.solve().  We recommend using
    the SCS solver.

    :param one_body_ints: spinless Fermion one-body integrals
    :param two_body_ints: spinless Fermion basis two-body integrals
    :param Int Na: number of Fermions with alpha-spin
    :param Int Nb: number of Fermions with beta-spin
    :return: cvxpy Problem object, variable dictionary
    :rtype: cvx.Problem
    """
    dim = one_body_ints.shape[0] // 2
    mm = dim ** 2
    mn = dim * (dim - 1) // 2

    h1a, h1b, v2aa, v2bb, v2ab = spin_adapted_interaction_tensor_rdm_consistent(two_body_ints,
                                                                               one_body_ints)
    print("constructing dual basis")
    dual_basis = sz_adapted_linear_constraints(dim, Na, Nb,
                                               ['ck', 'cckk', 'kkcc', 'ckck'])
    print("dual basis constructed")

    bas_aa, bas_ab = geminal_spin_basis(dim)

    v2ab.data *= 2.0

    copdm_a = h1a
    copdm_b = h1b
    coqdm_a = Tensor(np.zeros((dim, dim)), name='kc_a')
    coqdm_b = Tensor(np.zeros((dim, dim)), name='kc_b')
    ctpdm_aa = v2aa
    ctpdm_bb = v2bb
    ctpdm_ab = v2ab
    ctqdm_aa = Tensor(np.zeros((mn, mn)), name='kkcc_aa', basis=bas_aa)
    ctqdm_bb = Tensor(np.zeros((mn, mn)), name='kkcc_bb', basis=bas_aa)
    ctqdm_ab = Tensor(np.zeros((mm, mm)), name='kkcc_ab', basis=bas_ab)

    cphdm_ab = Tensor(np.zeros((dim, dim, dim, dim)), name='ckck_ab')
    cphdm_ba = Tensor(np.zeros((dim, dim, dim, dim)), name='ckck_ba')
    cphdm_aabb = Tensor(np.zeros((2 * mm, 2 * mm)), name='ckck_aabb')

    ctensor = MultiTensor([copdm_a, copdm_b, coqdm_a, coqdm_b, ctpdm_aa, ctpdm_bb, ctpdm_ab,
                           ctqdm_aa, ctqdm_bb, ctqdm_ab, cphdm_ab, cphdm_ba, cphdm_aabb])

    ctensor.dual_basis = dual_basis
    print('synthesizing dual basis')
    # A, _, b = ctensor.synthesize_dual_basis()
    print("dual basis synthesized")


    # create all the psd-matrices for the
    variable_dictionary = {}
    for tensor in ctensor.tensors:
        linear_dim = int(np.sqrt(tensor.size))
        variable_dictionary[tensor.name] = cvx.Variable(shape=(linear_dim, linear_dim), PSD=True, name=tensor.name)

    print("constructing constraints")
    constraints = []
    for dbe in dual_basis:
        single_constraint = []
        for tname, v_elements, p_coeffs in dbe:
            active_indices = get_var_indices(ctensor.tensors[tname], v_elements)
            # vec_idx = ctensor.tensors[tname].index_vectorized(*v_elements)
            # dim = int(np.sqrt(ctensor.tensors[tname].size))
            single_constraint.append(variable_dictionary[tname][active_indices] * p_coeffs)
        constraints.append(cvx.sum(single_constraint) == dbe.dual_scalar)
    print('constraints constructed')

    print("constructing the problem")
    # construct the problem variable for cvx
    objective = cvx.Minimize(
                cvx.trace(copdm_a.data * variable_dictionary['ck_a']) +
                cvx.trace(copdm_b.data * variable_dictionary['ck_b']) +
                cvx.trace(v2aa.data * variable_dictionary['cckk_aa']) +
                cvx.trace(v2bb.data * variable_dictionary['cckk_bb']) +
                cvx.trace(v2ab.data * variable_dictionary['cckk_ab']))

    cvx_problem = cvx.Problem(objective, constraints=constraints)
    print('problem constructed')
    return cvx_problem, variable_dictionary


def run_with_openfermion_cvxpy():
    # solve the spin problem
    import sys
    from openfermion.hamiltonians import MolecularData
    from openfermionpsi4 import run_psi4
    from openfermion.utils import map_one_pdm_to_one_hole_dm, map_two_pdm_to_two_hole_dm, map_two_pdm_to_particle_hole_dm
    from openfermion.transforms import jordan_wigner
    from representability.fermions.utils import get_molecule_openfermion
    from representability.fermions.constraints.test_antisymm_sz_constraints import system
    from representability.fermions.density.spin_density import SpinOrbitalDensity

    print('Running System Setup')
    basis = 'sto-3g'
    # basis = '6-31g'
    multiplicity = 0
    # charge = 0
    # geometry = [('H', [0.0, 0.0, 0.0]), ('H', [0, 0, 0.75])]
    # charge = 1
    # geometry = [('H', [0.0, 0.0, 0.0]), ('He', [0, 0, 0.75])]
    charge = 0
    bd = 1.2
    # geometry = [('H', [0.0, 0.0, 0.0]), ('H', [0, 0, bd]),
    #             ('H', [0.0, 0.0, 2 * bd]), ('H', [0, 0, 3 * bd])]
    geometry = [['H', [0, 0, 0]], ['H', [1.2, 0, 0]],
                ['H', [0, 1.2, 0]], ['H', [1.2, 1.2, 0]]]
    # geometry = [['He', [0, 0, 0]], ['H', [0, 0, 1.2]]]
    molecule = MolecularData(geometry, basis, multiplicity, charge)
    # Run Psi4.
    molecule = run_psi4(molecule,
                        run_scf=True,
                        run_mp2=False,
                        run_cisd=False,
                        run_ccsd=False,
                        run_fci=True,
                        delete_input=True)

    print('nuclear_repulsion', molecule.nuclear_repulsion)
    print('gs energy ', molecule.fci_energy)
    nuclear_repulsion = molecule.nuclear_repulsion
    gs_energy = molecule.fci_energy

    tpdm = np.einsum('ijkl->ijlk', molecule.fci_two_rdm)
    opdm = molecule.fci_one_rdm
    oqdm = map_one_pdm_to_one_hole_dm(opdm)
    tqdm = map_two_pdm_to_two_hole_dm(tpdm, opdm)
    phdm = map_two_pdm_to_particle_hole_dm(tpdm, opdm)

    tpdm = Tensor(tpdm, name='cckk')
    tqdm = Tensor(tqdm, name='kkcc')
    opdm = Tensor(opdm, name='ck')
    phdm = Tensor(phdm, name='ckck')

    hamiltonian = molecule.get_molecular_hamiltonian()
    one_body_ints, two_body_ints = hamiltonian.one_body_tensor, hamiltonian.two_body_tensor
    two_body_ints = np.einsum('ijkl->ijlk', two_body_ints)

    n_electrons = molecule.n_electrons
    print('n_electrons', n_electrons)
    Na = n_electrons // 2
    Nb = n_electrons // 2

    dim = one_body_ints.shape[0]
    mm = dim ** 2

    h1, v2 = spin_orbital_interaction_tensor(two_body_ints, one_body_ints)
    dual_basis = spin_orbital_linear_constraints(dim, Na, Nb,
                                                 ['ck', 'cckk', 'kkcc', 'ckck'],
                                                 sz=0)
    print("constructed dual basis")
    copdm = h1
    coqdm = Tensor(np.zeros((dim, dim)), name='kc')
    ctpdm = v2
    ctqdm = Tensor(np.zeros((dim, dim, dim, dim)), name='kkcc')
    cphdm = Tensor(np.zeros((dim, dim, dim, dim)), name='ckck')

    ctensor = MultiTensor([copdm, coqdm, ctpdm, ctqdm, cphdm])
    ctensor.dual_basis = dual_basis
    print("size of dual basis", len(dual_basis.elements))

    # create all the psd-matrices for the
    variable_dictionary = {}
    for tensor in ctensor.tensors:
        linear_dim = int(np.sqrt(tensor.size))
        variable_dictionary[tensor.name] = cvx.Variable(shape=(linear_dim, linear_dim), PSD=True, name=tensor.name)

    print("constructing constraints")
    constraints = []
    for dbe in dual_basis:
        single_constraint = []
        for tname, v_elements, p_coeffs in dbe:
            active_indices = get_var_indices(ctensor.tensors[tname], v_elements)
            single_constraint.append(variable_dictionary[tname][active_indices] * p_coeffs)
        constraints.append(cvx.sum(single_constraint) == dbe.dual_scalar)
    print('constraints constructed')

    print("constructing the problem")
    # construct the problem variable for cvx
    # interaction_integral_matrix = np.einsum('ijkl->ijlk', v2.data).reshape((dim**2, dim**2))
    interaction_integral_matrix = v2.data.reshape((dim**2, dim**2))

    objective = cvx.Minimize(
                cvx.trace(copdm.data @ variable_dictionary['ck']) +
                cvx.trace(interaction_integral_matrix @ variable_dictionary['cckk']))

    cvx_problem = cvx.Problem(objective, constraints=constraints)
    print('problem constructed')

    one_energy = np.trace(copdm.data.dot(opdm.data))
    two_energy = np.trace(interaction_integral_matrix @ tpdm.data.reshape((dim**2, dim**2)))  # np.einsum('ijkl,ijkl', tpdm.data, ctpdm.data)

    # cvx_problem.solve(solver=cvx.SCS, verbose=True, eps=0.5E-6, max_iters=60000)
    cvx_problem.solve(solver=cvx.SCS, verbose=True, eps=1.5E-5, max_iters=60000)
    # print(cvx_problem.value + nuclear_repulsion, gs_energy)
    # this should give something close to -2.147170020986181
    # assert np.isclose(cvx_problem.value + nuclear_repulsion, gs_energy)  # for 2-electron systems only
    print(variable_dictionary['cckk'].value)
    # np.save("HeHminus_631g_tpdm.npy", variable_dictionary['cckk'].value)
    print(cvx_problem.value + nuclear_repulsion)
    print(gs_energy)
    print(nuclear_repulsion)
    # assert np.isclose(cvx_problem.value + nuclear_repulsion, -2.147170020986181, rtol=1.0E-3)


def dqg_run_bpsdp():
    import sys
    from openfermion.hamiltonians import MolecularData
    from openfermionpsi4 import run_psi4
    from openfermionpyscf import run_pyscf
    from openfermion.utils import map_one_pdm_to_one_hole_dm, \
        map_two_pdm_to_two_hole_dm, map_two_pdm_to_particle_hole_dm

    print('Running System Setup')
    basis = 'sto-6g'
    # basis = '6-31g'
    multiplicity = 1
    # charge = 0
    # geometry = [('H', [0.0, 0.0, 0.0]), ('H', [0, 0, 0.75])]
    # charge = 1
    # geometry = [('H', [0.0, 0.0, 0.0]), ('He', [0, 0, 0.75])]
    charge = 0
    bd = 1.2
    # geometry = [('H', [0.0, 0.0, 0.0]), ('H', [0, 0, bd]),
    #             ('H', [0.0, 0.0, 2 * bd]), ('H', [0, 0, 3 * bd])]
    # geometry = [['H', [0, 0, 0]], ['H', [1.2, 0, 0]],
    #             ['H', [0, 1.2, 0]], ['H', [1.2, 1.2, 0]]]
    # geometry = [['He', [0, 0, 0]], ['H', [0, 0, 1.2]]]
    #  geometry = [['Be' [0, 0, 0]], [['B', [1.2, 0, 0]]]]
    geometry = [['N', [0, 0, 0]], ['N', [0, 0, 1.1]]]
    molecule = MolecularData(geometry, basis, multiplicity, charge)
    # Run Psi4.
    # molecule = run_psi4(molecule,
    #                     run_scf=True,
    #                     run_mp2=False,
    #                     run_cisd=False,
    #                     run_ccsd=False,
    #                     run_fci=True,
    #                     delete_input=True)
    molecule = run_pyscf(molecule,
                        run_scf=True,
                        run_mp2=False,
                        run_cisd=False,
                        run_ccsd=False,
                        run_fci=True)

    print('nuclear_repulsion', molecule.nuclear_repulsion)
    print('gs energy ', molecule.fci_energy)
    print("hf energy ", molecule.hf_energy)

    nuclear_repulsion = molecule.nuclear_repulsion
    gs_energy = molecule.fci_energy

    import openfermion as of
    hamiltonian = molecule.get_molecular_hamiltonian(occupied_indices=[0],
                                                     active_indices=[1,2,3,4])
    print(type(hamiltonian))
    print(hamiltonian)
    nuclear_repulsion = hamiltonian.constant
    hamiltonian.constant = 0
    ham = of.get_sparse_operator(hamiltonian).toarray()
    w, v = np.linalg.eigh(ham)
    idx = 0
    gs_energy = w[idx]
    n_density = v[:, [idx]] @ v[:, [idx]].conj().T

    from representability.fermions.density.antisymm_sz_density import AntiSymmOrbitalDensity

    density = AntiSymmOrbitalDensity(n_density, 8)
    opdm_a, opdm_b = density.construct_opdm()
    tpdm_aa, tpdm_bb, tpdm_ab, _ = density.construct_tpdm()

    true_tpdm = density.get_tpdm(density.rho, density.dim)
    true_tpdm = true_tpdm.transpose(0, 1, 3, 2)
    test_tpdm = unspin_adapt(tpdm_aa, tpdm_bb, tpdm_ab)
    assert np.allclose(true_tpdm, test_tpdm)

    tqdm_aa, tqdm_bb, tqdm_ab, _ = density.construct_thdm()
    phdm_ab, phdm_ba, phdm_aabb = density.construct_phdm()
    Na = np.round(opdm_a.trace()).real
    Nb = np.round(opdm_b.trace()).real

    one_body_ints, two_body_ints = hamiltonian.one_body_tensor, hamiltonian.two_body_tensor
    two_body_ints = np.einsum('ijkl->ijlk', two_body_ints)

    n_electrons = Na + Nb
    print('n_electrons', n_electrons)
    dim = one_body_ints.shape[0]
    spatial_basis_rank = dim // 2
    bij_bas_aa, bij_bas_ab = geminal_spin_basis(spatial_basis_rank)

    opdm_a_interaction, opdm_b_interaction, v2aa, v2bb, v2ab = \
        spin_adapted_interaction_tensor_rdm_consistent(two_body_ints,
                                                       one_body_ints)

    dual_basis = sz_adapted_linear_constraints(spatial_basis_rank, Na, Nb, ['ck', 'kc', 'cckk', 'ckck', 'kkcc'],
                                               S=1, M=-1)
    print("constructed dual basis")



    opdm_a = Tensor(opdm_a, name='ck_a')
    opdm_b = Tensor(opdm_b, name='ck_b')
    oqdm_a = Tensor(np.eye(dim//2) - opdm_a.data, name='kc_a')
    oqdm_b = Tensor(np.eye(dim//2) - opdm_b.data, name='kc_b')

    tpdm_aa = Tensor(tpdm_aa, name='cckk_aa', basis=bij_bas_aa)
    tpdm_bb = Tensor(tpdm_bb, name='cckk_bb', basis=bij_bas_aa)
    tpdm_ab = Tensor(tpdm_ab, name='cckk_ab', basis=bij_bas_ab)

    tqdm_aa = Tensor(tqdm_aa, name='kkcc_aa', basis=bij_bas_aa)
    tqdm_bb = Tensor(tqdm_bb, name='kkcc_bb', basis=bij_bas_aa)
    tqdm_ab = Tensor(tqdm_ab, name='kkcc_ab', basis=bij_bas_ab)

    phdm_ab = Tensor(phdm_ab, name='ckck_ab', basis=bij_bas_ab)
    phdm_ba = Tensor(phdm_ba, name='ckck_ba', basis=bij_bas_ab)
    phdm_aabb = Tensor(phdm_aabb, name='ckck_aabb')

    dtensor = MultiTensor([opdm_a, opdm_b, oqdm_a, oqdm_b,
                           tpdm_aa, tpdm_bb, tpdm_ab,
                           tqdm_aa, tqdm_bb, tqdm_ab,
                           phdm_ab, phdm_ba, phdm_aabb])

    copdm_a = opdm_a_interaction
    copdm_b = opdm_b_interaction
    coqdm_a = Tensor(np.zeros((spatial_basis_rank, spatial_basis_rank)), name='kc_a')
    coqdm_b = Tensor(np.zeros((spatial_basis_rank, spatial_basis_rank)), name='kc_b')
    ctpdm_aa = v2aa
    ctpdm_bb = v2bb
    ctpdm_ab = v2ab
    ctqdm_aa = Tensor(np.zeros_like(v2aa.data), name='kkcc_aa', basis=bij_bas_aa)
    ctqdm_bb = Tensor(np.zeros_like(v2bb.data), name='kkcc_bb', basis=bij_bas_aa)
    ctqdm_ab = Tensor(np.zeros_like(v2ab.data), name='kkcc_ab', basis=bij_bas_ab)
    cphdm_ab = Tensor(np.zeros((spatial_basis_rank**2, spatial_basis_rank**2)),
                      name='ckck_ab', basis=bij_bas_ab)
    cphdm_ba = Tensor(np.zeros((spatial_basis_rank**2, spatial_basis_rank**2)),
                      name='ckck_ba', basis=bij_bas_ab)
    cphdm_aabb = Tensor(np.zeros((2 * spatial_basis_rank**2, 2 * spatial_basis_rank**2)),
                        name='ckck_aabb')

    ctensor = MultiTensor([copdm_a, copdm_b, coqdm_a, coqdm_b, ctpdm_aa, ctpdm_bb, ctpdm_ab, ctqdm_aa, ctqdm_bb,
                           ctqdm_ab, cphdm_ab, cphdm_ba, cphdm_aabb])

    print((ctensor.vectorize_tensors().T @ dtensor.vectorize_tensors())[0, 0].real)
    print(gs_energy)

    ctensor.dual_basis = dual_basis
    A, _, b = ctensor.synthesize_dual_basis()
    print("size of dual basis", len(dual_basis.elements))

    print(A @ dtensor.vectorize_tensors() - b)

    nc, nv = A.shape
    A.eliminate_zeros()
    nnz = A.nnz

    from sdpsolve.sdp import SDP
    from sdpsolve.solvers.bpsdp import solve_bpsdp
    from sdpsolve.solvers.bpsdp.bpsdp_old import solve_bpsdp
    from sdpsolve.utils.matreshape import vec2block
    sdp = SDP()

    sdp.nc = nc
    sdp.nv = nv
    sdp.nnz = nnz
    sdp.blockstruct = list(map(lambda x: int(np.sqrt(x.size)), ctensor.tensors))
    sdp.nb = len(sdp.blockstruct)
    sdp.Amat = A.real
    sdp.bvec = b.todense().real
    sdp.cvec = ctensor.vectorize_tensors().real

    sdp.Initialize()
    epsilon = 1.0E-7
    sdp.epsilon = float(epsilon)
    sdp.epsilon_inner = float(epsilon) / 100

    sdp.disp = True
    sdp.iter_max = 70000
    sdp.inner_solve = 'CG'
    sdp.inner_iter_max = 2

    # # sdp_data = solve_bpsdp(sdp)
    solve_bpsdp(sdp)
    # # create all the psd-matrices for the
    # variable_dictionary = {}
    # for tensor in ctensor.tensors:
    #     linear_dim = int(np.sqrt(tensor.size))
    #     variable_dictionary[tensor.name] = cvx.Variable(shape=(linear_dim, linear_dim), PSD=True, name=tensor.name)

    # print("constructing constraints")
    # constraints = []
    # for dbe in dual_basis:
    #     single_constraint = []
    #     for tname, v_elements, p_coeffs in dbe:
    #         active_indices = get_var_indices(ctensor.tensors[tname], v_elements)
    #         single_constraint.append(variable_dictionary[tname][active_indices] * p_coeffs)
    #     constraints.append(cvx.sum(single_constraint) == dbe.dual_scalar)
    # print('constraints constructed')

    # print("constructing the problem")
    # objective = cvx.Minimize(
    #             cvx.trace(copdm_a.data @ variable_dictionary['ck_a']) +
    #             cvx.trace(copdm_b.data @ variable_dictionary['ck_b']) +
    #             cvx.trace(ctpdm_aa.data @ variable_dictionary['cckk_aa']) +
    #             cvx.trace(ctpdm_bb.data @ variable_dictionary['cckk_bb']) +
    #             cvx.trace(ctpdm_ab.data @ variable_dictionary['cckk_ab']))

    # cvx_problem = cvx.Problem(objective, constraints=constraints)
    # print('problem constructed')

    # cvx_problem.solve(solver=cvx.SCS, verbose=True, eps=0.5E-5, max_iters=100000)


    # rdms_solution = vec2block(sdp.blockstruct, sdp.primal)

    print(gs_energy)
    # print(cvx_problem.value + nuclear_repulsion)
    # print(sdp_data.primal_value() + nuclear_repulsion)
    print(sdp.primal.T @ sdp.cvec)

    print(nuclear_repulsion)
    rdms = vec2block(sdp.blockstruct, sdp.primal)


    tpdm = unspin_adapt(rdms[4], rdms[5], rdms[6])
    print(np.einsum('ijij', tpdm))
    tpdm = np.einsum('ijkl->ijlk', tpdm)
    # np.save("h4_sto6g_1.2A_DQG_so_tpdm", tpdm)



def v2rdm_hubbard():
    import sys
    from openfermion.hamiltonians import MolecularData
    from openfermionpsi4 import run_psi4
    from openfermionpyscf import run_pyscf
    from openfermion.utils import map_one_pdm_to_one_hole_dm, \
        map_two_pdm_to_two_hole_dm, map_two_pdm_to_particle_hole_dm
    import openfermion as of

    e_fci = []
    e_rdm = []
    for U in [4]: # range(1, 11):
        sites = 5
        hubbard = of.hamiltonians.fermi_hubbard(1, sites, tunneling=1, coulomb=U,
                                                chemical_potential=0,
                                                magnetic_field=0,
                                                periodic=True,
                                                spinless=False)
        # op_mat = of.get_sparse_operator(hubbard).toarray()
        # # op_mat = of.get_number_preserving_sparse_operator(hubbard, sites * 2, sites-1).toarray()
        # w, v = np.linalg.eigh(op_mat)
        # # w_idx = 5 # N4U4
        # # w_idx = 25  # N6 U4
        # w_idx = 4
        # n_density = v[:, [w_idx]] @ v[:, [w_idx]].conj().T
        # from representability.fermions.density.antisymm_sz_density import AntiSymmOrbitalDensity

        # density = AntiSymmOrbitalDensity(n_density, sites * 2)
        # tpdm_aa, tpdm_bb, tpdm_ab, [bas_aa, bas_ab] = density.construct_tpdm()
        # rev_bas_aa = dict(zip(bas_aa.values(), bas_aa.keys()))
        # rev_bas_ab = dict(zip(bas_ab.values(), bas_ab.keys()))
        # for r, s in product(range(sites), repeat=2):
        #     i, j = rev_bas_ab[r]
        #     k, l = rev_bas_ab[s]
        # tqdm_aa, tqdm_bb, tqdm_ab, _ = density.construct_thdm()
        # phdm_ab, phdm_ba, phdm_aabb = density.construct_phdm()
        # opdm_a, opdm_b = density.construct_opdm()
        # bas_aa, bas_ab = geminal_spin_basis(sites)

        # opdm_a = Tensor(opdm_a, name='ck_a')
        # opdm_b = Tensor(opdm_b, name='ck_b')
        # oqdm_a = Tensor(np.eye(4) - opdm_a.data, name='kc_a')
        # oqdm_b = Tensor(np.eye(4) - opdm_b.data, name='kc_b')
        # tpdm_aa = Tensor(tpdm_aa, name='cckk_aa', basis=bas_aa)
        # tpdm_bb = Tensor(tpdm_bb, name='cckk_bb', basis=bas_aa)
        # tpdm_ab = Tensor(tpdm_ab, name='cckk_ab', basis=bas_ab)
        # tqdm_aa = Tensor(tqdm_aa, name='kkcc_aa', basis=bas_aa)
        # tqdm_bb = Tensor(tqdm_bb, name='kkcc_bb', basis=bas_aa)
        # tqdm_ab = Tensor(tqdm_ab, name='kkcc_ab', basis=bas_ab)
        # phdm_ab = Tensor(phdm_ab, name='ckck_ab', basis=bas_ab)
        # phdm_ba = Tensor(phdm_ba, name='ckck_ba', basis=bas_ab)
        # phdm_aabb = Tensor(phdm_aabb, name='ckck_aabb')
        # rdms = MultiTensor(
        #     [opdm_a, opdm_b, oqdm_a, oqdm_b,
        #      tpdm_aa, tpdm_bb, tpdm_ab,
        #      tqdm_aa, tqdm_bb, tqdm_ab,
        #      phdm_ab, phdm_ba, phdm_aabb])
        # rdmvec = rdms.vectorize_tensors()

        hamiltonian = of.get_interaction_operator(hubbard)
        op_mat = of.get_number_preserving_sparse_operator(hubbard, 2 * sites, sites - 1, spin_preserving=False).toarray()
        w, _ = np.linalg.eigh(op_mat)

        gs_e = w[0]
        print(gs_e)

        one_body_ints, two_body_ints = hamiltonian.one_body_tensor, hamiltonian.two_body_tensor
        two_body_ints = np.einsum('ijkl->ijlk', two_body_ints)

        n_electrons = sites - 1
        print('n_electrons', n_electrons)
        Na = n_electrons // 2
        Nb = n_electrons // 2
        dim = one_body_ints.shape[0]
        spatial_basis_rank = sites
        sdim = spatial_basis_rank
        mm = dim ** 2
        bij_bas_aa, bij_bas_ab = geminal_spin_basis(spatial_basis_rank)

        # h1, v2 = spin_orbital_interaction_tensor(two_body_ints, one_body_ints)

        opdm_a_interaction, opdm_b_interaction, v2aa, v2bb, v2ab = \
            spin_adapted_interaction_tensor_rdm_consistent(two_body_ints.real,
                                                           one_body_ints.real)

        v2ab_mat = np.zeros_like(v2ab.data)
        for i in range(spatial_basis_rank):
            # ia^ j^b j^b ia
            idx = bij_bas_ab.rev((i, i))
            v2ab_mat[idx, idx] = U

        v2ab = Tensor(v2ab_mat, basis=v2ab.basis, name=v2ab.name)

        dual_basis = sz_adapted_linear_constraints(spatial_basis_rank, Na, Nb,
                                                   ['ck', 'cckk', 'kkcc', 'ckck'])

        print("constructed dual basis")

        copdm_a = opdm_a_interaction
        copdm_b = opdm_b_interaction
        coqdm_a = Tensor(np.zeros((spatial_basis_rank, spatial_basis_rank)),
                         name='kc_a')
        coqdm_b = Tensor(np.zeros((spatial_basis_rank, spatial_basis_rank)),
                         name='kc_b')
        ctpdm_aa = v2aa
        ctpdm_bb = v2bb
        ctpdm_ab = v2ab
        ctqdm_aa = Tensor(np.zeros_like(v2aa.data), name='kkcc_aa',
                          basis=bij_bas_aa)
        ctqdm_bb = Tensor(np.zeros_like(v2bb.data), name='kkcc_bb',
                          basis=bij_bas_aa)
        ctqdm_ab = Tensor(np.zeros_like(v2ab.data), name='kkcc_ab',
                          basis=bij_bas_ab)
        cphdm_ab = Tensor(np.zeros((spatial_basis_rank * spatial_basis_rank,
                                    spatial_basis_rank * spatial_basis_rank)),
                          name='ckck_ab', basis=bij_bas_ab)
        cphdm_ba = Tensor(np.zeros((spatial_basis_rank * spatial_basis_rank,
                                    spatial_basis_rank * spatial_basis_rank)),
                          name='ckck_ba', basis=bij_bas_ab)
        cphdm_aabb = Tensor(
            np.zeros((2 * spatial_basis_rank ** 2, 2 * spatial_basis_rank ** 2)),
            name='ckck_aabb')

        ctensor = MultiTensor(
            [copdm_a, copdm_b, coqdm_a, coqdm_b, ctpdm_aa, ctpdm_bb, ctpdm_ab,
             ctqdm_aa, ctqdm_bb, ctqdm_ab,
             cphdm_ab, cphdm_ba, cphdm_aabb])

        ctensor.dual_basis = dual_basis
        A, _, b = ctensor.synthesize_dual_basis()
        print("size of dual basis", len(dual_basis.elements))

        # print(tpdm_ab.data.trace())
        # print(ctensor.vectorize_tensors().T @ rdmvec)
        # print(b.shape)
        # print("FCI Residual ", np.linalg.norm(A @ rdmvec - b))
        # exit()

        nc, nv = A.shape
        # A.eliminate_zeros()
        nnz = A.nnz

        sdp = SDP()
        sdp.nc = nc
        sdp.nv = nv
        sdp.nnz = nnz
        sdp.blockstruct = list(map(lambda x: int(np.sqrt(x.size)), ctensor.tensors))
        sdp.nb = len(sdp.blockstruct)
        sdp.Amat = A.real
        sdp.bvec = b.todense().real
        sdp.cvec = ctensor.vectorize_tensors().real

        Amat = A.toarray()
        # print(A.shape)
        # # DQ num vars: 4, 4, 4, 4, 6, 6, 16, 6, 6 16
        # print("D2 size ", sum([x**2 for x in [6, 6, 16]]))
        # print("D1Q1 size ", sum([x**2 for x in [4, 4, 4, 4]]))
        # print("Spin constraint ", 16**2)
        # print(sum([x**2 for x in [4, 4, 4, 4, 6, 6, 16, 16]]))

        sm = sdim * (sdim - 1) // 2

        uadapt = gen_trans_2rdm(sdim**2, sdim)

        # spin_adapted_d2ab = uadapt.T @ tpdm_ab.data @ uadapt
        # d2ab_sa_a = spin_adapted_d2ab[:sm, :sm]
        # d2ab_sa_s = spin_adapted_d2ab[sm:, sm:]

        # for r, s in product(range(sdim * (sdim - 1) // 2), repeat=2):
        #     i, j = bas_aa.fwd(r)
        #     k, l = bas_aa.fwd(s)
        #     # print((i, j, k, l), d2ab_sa_a[bas_aa.rev((i, j)), bas_aa.rev((k, l))],
        #     #       uadapt[:, r].T @ tpdm_ab.data @ uadapt[:, s]
        #     #       )
        #     assert np.isclose(d2ab_sa_a[bas_aa.rev((i, j)), bas_aa.rev((k, l))], uadapt[:, [r]].T @ tpdm_ab.data @ uadapt[:, [s]])
        #     assert np.isclose(d2ab_sa_a[bas_aa.rev((i, j)), bas_aa.rev((k, l))], np.trace(tpdm_ab.data @ (uadapt[:, [s]] @ uadapt[:, [r]].T)))
        #     assert np.isclose(d2ab_sa_a[bas_aa.rev((i, j)), bas_aa.rev((k, l))], np.einsum('ij,ij', tpdm_ab.data, (uadapt[:, [s]] @ uadapt[:, [r]].T)))
        #     assert np.isclose(tpdm_aa.data[r, s] + tpdm_aa.data[s, r], uadapt[:, [r]].T @ tpdm_ab.data @ uadapt[:, [s]] + uadapt[:, [s]].T @ tpdm_ab.data @ uadapt[:, [r]])
        #     assert np.isclose(tpdm_bb.data[r, s] + tpdm_bb.data[s, r], uadapt[:, [r]].T @ tpdm_ab.data @ uadapt[:, [s]] + uadapt[:, [s]].T @ tpdm_ab.data @ uadapt[:, [r]])

        print("AA Dim: ", sdim * (sdim - 1) / 2, sm * (sm + 1) / 2)
        for ii in range(Amat.shape[0]):
            amats = vec2block(sdp.blockstruct, Amat[ii, :])
            for aa in amats:
                assert of.is_hermitian(aa)


        sdp.Initialize()
        epsilon = 1.0E-6
        sdp.epsilon = float(epsilon)
        sdp.epsilon_inner = float(epsilon)
        sdp.disp = True
        sdp.iter_max = 50000
        sdp.inner_iter_max = 1
        sdp.inner_solve = 'CG'

        write_sdpfile("new_hubbardN{}U{}_DQG.sdp".format(sites, U), sdp.nc, sdp.nv, sdp.nnz, sdp.nb, sdp.Amat,
                      sdp.bvec, sdp.cvec, sdp.blockstruct)
        # sdp_data = solve_bpsdp(sdp)
        # sdp_data.primal_vector = rdmvec
        # sdp.iter_max = 5000
        #  sdp_data = solve_bpsdp(sdp)
        solve_rrsdp(sdp)
        print(sdp.primal.T @ sdp.cvec, gs_e)


def v2rdm_hubbard_open_shell():
    import openfermion as of

    e_fci = []
    e_rdm = []
    for U in [1]: # range(1, 11):
        sites = 5
        hubbard = of.hamiltonians.fermi_hubbard(1, sites, tunneling=1, coulomb=U,
                                                chemical_potential=0,
                                                magnetic_field=0,
                                                periodic=True,
                                                spinless=False)
        hamiltonian = of.get_interaction_operator(hubbard)
        op_mat = of.get_number_preserving_sparse_operator(hubbard, 2 * sites,
                                                          sites,
                                                          spin_preserving=True).toarray()
        sz_mat = of.get_number_preserving_sparse_operator(of.sz_operator(sites),
                                                          2 * sites, sites,
                                                          spin_preserving=True).toarray()
        s2_mat = of.get_number_preserving_sparse_operator(of.s_squared_operator(sites),
                                                          2 * sites, sites,
                                                          spin_preserving=True).toarray()

        w, v = np.linalg.eigh(op_mat)
        sz_exp = []
        s2_exp = []
        for ii in range(len(w)):
            sz_exp.append((v[:, [ii]].conj().T @ sz_mat @ v[:, [ii]])[0, 0].real)
            s2_exp.append((v[:, [ii]].conj().T @ s2_mat @ v[:, [ii]])[0, 0].real)

        print(sz_exp[:10])
        print(s2_exp[:10])
        print(w[:10])

        gs_e = w[0]
        print(gs_e)

        one_body_ints, two_body_ints = hamiltonian.one_body_tensor, hamiltonian.two_body_tensor
        two_body_ints = np.einsum('ijkl->ijlk', two_body_ints)

        n_electrons = sites
        print('n_electrons', n_electrons)
        Na = 1 + (n_electrons // 2)
        Nb = n_electrons // 2
        spatial_basis_rank = sites
        bij_bas_aa, bij_bas_ab = geminal_spin_basis(spatial_basis_rank)

        opdm_a_interaction, opdm_b_interaction, v2aa, v2bb, v2ab = \
            spin_adapted_interaction_tensor_rdm_consistent(two_body_ints.real,
                                                           one_body_ints.real)

        v2ab_mat = np.zeros_like(v2ab.data)
        for i in range(spatial_basis_rank):
            # ia^ j^b j^b ia
            idx = bij_bas_ab.rev((i, i))
            v2ab_mat[idx, idx] = U

        v2ab = Tensor(v2ab_mat, basis=v2ab.basis, name=v2ab.name)

        dual_basis = sz_adapted_linear_constraints(spatial_basis_rank, Na, Nb,
                                                   ['ck', 'cckk', 'kkcc', 'ckck'],
                                                   S=0.5, M=0.5)

        print("constructed dual basis")

        copdm_a = opdm_a_interaction
        copdm_b = opdm_b_interaction
        coqdm_a = Tensor(np.zeros((spatial_basis_rank, spatial_basis_rank)),
                         name='kc_a')
        coqdm_b = Tensor(np.zeros((spatial_basis_rank, spatial_basis_rank)),
                         name='kc_b')
        ctpdm_aa = v2aa
        ctpdm_bb = v2bb
        ctpdm_ab = v2ab
        ctqdm_aa = Tensor(np.zeros_like(v2aa.data), name='kkcc_aa',
                          basis=bij_bas_aa)
        ctqdm_bb = Tensor(np.zeros_like(v2bb.data), name='kkcc_bb',
                          basis=bij_bas_aa)
        ctqdm_ab = Tensor(np.zeros_like(v2ab.data), name='kkcc_ab',
                          basis=bij_bas_ab)
        cphdm_ab = Tensor(np.zeros((spatial_basis_rank * spatial_basis_rank,
                                    spatial_basis_rank * spatial_basis_rank)),
                          name='ckck_ab', basis=bij_bas_ab)
        cphdm_ba = Tensor(np.zeros((spatial_basis_rank * spatial_basis_rank,
                                    spatial_basis_rank * spatial_basis_rank)),
                          name='ckck_ba', basis=bij_bas_ab)
        cphdm_aabb = Tensor(
            np.zeros((2 * spatial_basis_rank ** 2, 2 * spatial_basis_rank ** 2)),
            name='ckck_aabb')

        ctensor = MultiTensor(
            [copdm_a, copdm_b, coqdm_a, coqdm_b, ctpdm_aa, ctpdm_bb, ctpdm_ab,
             ctqdm_aa, ctqdm_bb, ctqdm_ab,
             cphdm_ab, cphdm_ba, cphdm_aabb])

        ctensor.dual_basis = dual_basis
        A, _, b = ctensor.synthesize_dual_basis()
        print("size of dual basis", len(dual_basis.elements))

        nc, nv = A.shape
        nnz = A.nnz

        sdp = SDP()
        sdp.nc = nc
        sdp.nv = nv
        sdp.nnz = nnz
        sdp.blockstruct = list(map(lambda x: int(np.sqrt(x.size)), ctensor.tensors))
        sdp.nb = len(sdp.blockstruct)
        sdp.Amat = A.real
        sdp.bvec = b.todense().real
        sdp.cvec = ctensor.vectorize_tensors().real

        sdp.Initialize()
        epsilon = 0.5e-5
        sdp.epsilon = float(epsilon)
        sdp.epsilon_inner = float(epsilon)
        sdp.disp = True
        sdp.iter_max = 50000
        sdp.inner_iter_max = 1
        sdp.inner_solve = 'CG'
        sdp_data = solve_bpsdp(sdp)
        print(sdp.primal.T @ sdp.cvec, gs_e)

if __name__ == "__main__":
    # dqg_run_bpsdp()
    # v2rdm_hubbard()
    v2rdm_hubbard_open_shell()
    print("HRE")
