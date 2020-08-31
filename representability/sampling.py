from itertools import product
import numpy as np


def add_gaussian_noise(tensor, sqrt_variance):
    """
    Iterate over tensor and apply Gaussian noise

    :param tensor:
    :param M:
    :param variance:
    :return:
    """
    corrupted_tensor = np.copy(tensor)
    if np.isclose(sqrt_variance, 0.0):
        return corrupted_tensor

    for indices in product(range(tensor.shape[0]), repeat=tensor.ndim):
        corrupted_tensor[indices] += np.random.normal(0, scale=sqrt_variance)

    return corrupted_tensor


def add_gaussian_noise_antisymmetric_four_tensor(tensor, std_error):
    """
    Iterate over tensor and apply Gaussian noise

    :param tensor:
    :param M:
    :param std_error:
    :return:
    """
    corrupted_tensor = np.copy(tensor)
    if np.isclose(std_error, 0.0):
        return corrupted_tensor

    dim = tensor.shape[0]
    for p, q, r, s in product(range(dim), repeat=4):
        if p * dim + q >= r * dim + s and p < q and r < s:
            # the normal distribution used by numpy, scale variable is the dtandard error
            # which is the square root of the variance
            corrupted_tensor[p, q, r, s] += np.random.normal(0, scale=std_error)
            corrupted_tensor[q, p, r, s] = -corrupted_tensor[p, q, r, s]
            corrupted_tensor[p, q, s, r] = -corrupted_tensor[p, q, r, s]
            corrupted_tensor[q, p, s, r] = corrupted_tensor[p, q, r, s]

            corrupted_tensor[r, s, p, q] = corrupted_tensor[p, q, r, s]
            corrupted_tensor[r, s, q, p] = corrupted_tensor[q, p, r, s]
            corrupted_tensor[s, r, p, q] = corrupted_tensor[p, q, s, r]
            corrupted_tensor[s, r, q, p] = corrupted_tensor[q, p, s, r]

    return corrupted_tensor


def parity_even_p(state, marked_qubits):
    """
    Calculates the parity of elements at indexes in marked_qubits

    Parity is relative to the binary representation of the integer state.

    :param state: The wavefunction index that corresponds to this state.
    :param marked_qubits: The indexes to be considered in the parity sum.
    :returns: A boolean corresponding to the parity.
    """
    assert isinstance(state, int), "{} is not an integer. Must call " \
                                   "parity_even_p with an integer " \
                                   "state.".format(state)
    mask = 0
    for q in marked_qubits:
        mask |= 1 << q
    return bin(mask & state).count("1") % 2 == 0
