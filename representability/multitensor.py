import numpy as np
from scipy.sparse import csr_matrix
from representability.dualbasis import DualBasis, DualBasisElement


class TMap(object):
    def __init__(self, tensors):
        """
        provide a map of tensor name to tensors

        :param tensors: list of tensors
        :return: TMap object
        """
        self.tensors = tensors

    def __getitem__(self, i):
        # ew...I get remade everytime.
        _map = dict(map(lambda x: (x.name, x), self.tensors))
        return _map[i]

    def __iter__(self):
        for tt in self.tensors:
            yield tt


class MultiTensor(object):

    def __init__(self, tensors, dual_basis=DualBasis()):
        """
        A collection of tensor objects with algebraic maps from tensor to tensor

        In order to define a linear relationship between two tensors--i.e. the opdm
        and the oqdm-- a collection of maps objects can be generated.

        Mathematically, this is an object that allows you to define the dual basis on
        the vector space defined by the direct sum of all the tensors

        :param tensors: a dictionary or tuple of tensors and their associated call name
        :param DualBasisElement dual_basis:
        """
        if not isinstance(tensors, list):
            raise TypeError("MultiTensor accepts a list")

        # this preserves the order the user passes with the tensors
        self.tensors = TMap(tensors)

        # since all the tensors are indexed from zero...I need to know their 
        # numbering offset when combined with everything.
        self.off_set_map = self.make_offset_dict(self.tensors)

        # An iterable object that provides access to the dual basis elements
        self.dual_basis = dual_basis
        self.vec_dim = sum([vec.size for vec in self.tensors])

    @staticmethod
    def make_offset_dict(tensors):
        if not isinstance(tensors, TMap):
            raise TypeError("Tensors must be a TMap")

        tally = 0
        offsets = {}
        # this is why ordered dicts are great.  Remembering orderings
        for tensor_value in tensors:
            offsets[tensor_value.name] = tally
            tally += tensor_value.size

        return offsets

    def vectorize_tensors(self):
        """
        vectorize the tensors

        :returns: a vectorized form of the tensors
        """
        vec = np.empty((0, 1))
        for tensor in self.tensors:
            vec = np.vstack((vec, tensor.vectorize()))
        return vec

    def add_dual_elements(self, dual_element):
        """
        Add  a dual element to the dual basis
        """
        if not isinstance(dual_element, DualBasisElement):
            raise TypeError("dual_element variable needs to be a DualBasisElement type")

        # we should extend TMap to add
        self.dual_basis.elements.extend(dual_element)

    def synthesize_dual_basis(self):
        """
        from the list of maps create a m x n sparse matrix for Ax=b

        where x is the vectorized form of all the tensors.  This would be
        the very last step usually.

        :returns: sparse matrix
        """
        # go throught the dual basis list and synthesize each element
        dual_row_indices = []
        dual_col_indices = []
        dual_data_values = []

        # this forms the b-vector of ax + b = c
        bias_data_values = []
        # this forms the c-vector of ax + b = c
        inner_prod_data_values = []

        for index, dual_element in enumerate(self.dual_basis):
            dcol, dval = self.synthesize_element(dual_element)
            dual_row_indices.extend([index] * len(dcol))
            dual_col_indices.extend(dcol)
            dual_data_values.extend(dval)
            inner_prod_data_values.append(dual_element.dual_scalar)
            bias_data_values.append(dual_element.constant_bias)

        sparse_dual_operator = csr_matrix((dual_data_values,
                                          (dual_row_indices, dual_col_indices)),
                                          [index + 1, self.vec_dim])

        sparse_bias_vector = csr_matrix((bias_data_values,
                                        (range(index + 1), [0] * (index + 1))),
                                        [index + 1, 1])

        sparse_innerp_vector = csr_matrix((inner_prod_data_values,
                                           (range(index + 1), [0] * (index + 1))),
                                           [index + 1, 1])

        return sparse_dual_operator, sparse_bias_vector, sparse_innerp_vector

    def synthesize_element(self, element):
        """
        Generate the row index and column index for an element

        :param DualBasisElement element: element of the dual basis to vectorize
        """
        col_idx = []
        data_vals = []
        for tlabel, velement, coeff in element:
            col_idx.append(self.off_set_map[tlabel] + self.tensors[tlabel].index_vectorized(*velement))
            data_vals.append(coeff)

        return col_idx, data_vals
