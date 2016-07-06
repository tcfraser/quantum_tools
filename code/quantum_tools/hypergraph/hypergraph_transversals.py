from ..examples.symbolic_contexts import *
from ..symmetries.workspace import *
from ..visualization.sparse_vis import plot_coo_matrix
from ..config import *
from scipy import sparse
import numpy as np
from functools import reduce
from operator import mul
import inspect

def nzcfr(M, row):
    """ Non zero columns for a given row of a csr matrix """
    return M.indices[M.indptr[row]:M.indptr[row+1]]

def ptrflux(indptr):
    flux = indptr[1:] - indptr[:-1]
    return np.nonzero(flux)[0]

def sparse_any_eq(A, B):
    """
    SparseEfficiencyWarning: Comparing sparse matrices using == is inefficient, try using != instead.
    """
    # If anything in the sparse diff is false, then some of A and B are equal
    sparse_diff = A != B
    diff_len = reduce(mul, sparse_diff.shape, 1)

    if sparse_diff.nnz == diff_len:
        return False
    else:
        return True

def hyper_graph(A, ant):
    A_coo = sparse.coo_matrix(A)
    A_csr = sparse.csr_matrix(A)
    # A_csc = sparse.csc_matrix(A)
    # Take only columns that are extendable versions of the antescedant (ant)
    ext_ant = nzcfr(A_csr, ant)
    ext_A = A_csr[:, ext_ant]
    # Take only rows that would contribute
    row_entry_flux = ext_A.indptr[1:] - ext_A.indptr[:-1] # Finds where the row pointers change
    row_entry_flux[ant] = 0 # Make antecedant zero so it doesn't get selected
    H = ext_A[np.nonzero(row_entry_flux)[0], :]
    return H

class VerboseLog():

    def __init__(self, on=False):
        self.on = on

    def log(self, *args):
        if self.on:
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
            caller_name = calframe[1][3]
            print('[{name}]:'.format(name=caller_name), *args)

EXP_DTYPE = 'int16' # Can't be bool. Scipy sparse arrays don't support boolean matrices well

class FoundTransversals():

    def __init__(self, strat):
        self._raw = None
        self._raw_sum = None
        self._strat = strat

    def __len__(self):
        if self._raw is None:
            return 0
        else:
            return self._raw.shape[1]

    def update(self, ft):
        ft = sparse.csc_matrix(ft)
        total_ft = len(ft.data) # All identity elements
        if self._raw is None:
            # start it off
            self._raw = sparse.csc_matrix(ft)
            self._raw_sum = np.array([total_ft], dtype=EXP_DTYPE)
        else:
            # need to check against found transversals
            overlap = _overlap(self._raw, ft)
            if np.any(overlap == self._raw_sum): # there is a found transversal more minimal than ft
                return
            fts_to_keep = overlap != total_ft # Keep those that are not supersets of ft

            self._raw = sparse.hstack((self._raw[:, fts_to_keep], ft))
            self._raw_sum = np.append(self._raw_sum[fts_to_keep], total_ft)
        return self._strat.is_max_transversal(len(self))

    def minimal_present(self, ft):
        if self._raw is None:
            return False
        else:
            overlap = _overlap(self._raw, ft)
            return np.any(overlap == self._raw_sum) # There is a more minimal one present

    def __getitem__(self, slice):
        if self._raw is None:
            return None
        return self._raw[slice]

    def raw(self):
        return self._raw

def _overlap(a, b):
    return (a.T * b).toarray().ravel()

def is_minimal_present(t, ts):
    """ Determines if a more mimimal transversal than t is present in ts """
    if ts is None:
        return False
    overlap = _overlap(ts, t) # How much t overlaps with the elements of ts
    total_fs = _overlap(ts, get_full_transversal(t.shape[0])) # Sums over the columns
    any_minimal = np.any(overlap == total_fs)
    return any_minimal

def cernikov_filter(wts, fts=None):
    """ Remove any of the working transversals that are minimal versions of each other """
    wt_i = 0
    wts = sparse.csc_matrix(wts)
    vl.log('Filtering {0}...'.format(wts.shape[1]))
    if fts is not None:
        fts = sparse.csc_matrix(fts)

    while wt_i < wts.shape[1]:
        target_t = wts[:, wt_i]
        left_t = wts[:, :max(wt_i, 0)]
        right_t = wts[:, min(wt_i+1, wts.shape[1]):]

        assert(left_t.shape[1] + right_t.shape[1] + target_t.shape[1] == wts.shape[1]), "Left/Right split failed."

        left_right_t = sparse.hstack((left_t, right_t))
        if fts is not None:
            check_ts = sparse.hstack((left_right_t, fts))
        else:
            check_ts = left_right_t

        if is_minimal_present(target_t, check_ts):
            wts = left_right_t # The new wts to loop over
            # [logic] wt_i = wt_i # Don't increase
        else:
            # [logic] wts = wts # Keep target
            wt_i += 1
    vl.log('...down to {0}'.format(wts.shape[1]))
    return wts

class HTStrat():

    def __init__(self, search=None, max_t=None):
        self._search = search
        self._max_t = max_t

    def get_worker(self):
        if self._search is None:
            return work_on_transversals_depth
        if self._search == 'depth':
            return work_on_transversals_depth
        if self._search == 'breadth':
            return work_on_transversals_breadth
        raise Exception("Invalid search strat.")

    def is_max_transversal(self, count):
        if self._max_t is None:
            return False
        else:
            return count >= self._max_t

def get_null_transveral(size):
    """
    size : number of nodes
    """
    return sparse.csc_matrix((size, 1), dtype=EXP_DTYPE)

_full_transversal_pool = {}
def get_full_transversal(size):
    """
    size : number of nodes
    """
    global _full_transversal_pool
    if size not in _full_transversal_pool:
        _full_transversal_pool[size] = sparse.csc_matrix(np_ones((size, 1)), dtype=EXP_DTYPE)
    return _full_transversal_pool[size]

_unit_completion_pool = {}
def get_unit_completion(size):
    """
    size : number of edges
    """
    global _unit_completion_pool
    if size not in _unit_completion_pool:
        _unit_completion_pool[size] = sparse.csr_matrix(np_ones(size), dtype=EXP_DTYPE)
    return _unit_completion_pool[size]

_numpy_ones_pool = {}
def np_ones(shape):
    """
    shape: The numpy of ones
    """
    global _numpy_ones_pool
    if shape not in _numpy_ones_pool:
        _numpy_ones_pool[shape] = np.ones(shape, dtype=EXP_DTYPE)
    return _numpy_ones_pool[shape]

def find_transversals(H, strat=None):
    strat = HTStrat() if strat is None else strat
    H = sparse.csc_matrix(H) # Needed for the algorithm
    fts = FoundTransversals(strat) # Empty found transversals
    num_nodes = H.shape[0]
    strat.get_worker()(H, get_null_transveral(num_nodes), fts)
    return fts

def verify_completion(H, wt, fts):
    # Check if transversal is complete
    completion = transversal_completion(H, wt)
    vl.log('completion')
    vl.log(completion)
    vl.log('is_completion')
    vl.log(is_complete_transversal(completion))
    if is_complete_transversal(completion):
        fts_response = fts.update(wt) # Update the list of transversals with the new found one
        return True, None, fts_response
    return False, completion, False # Haven't hit max iterations

def work_on_transversals_breadth(H, wts, fts, current_depth=0):
    """ Work on a particular transversal going breadth first """
    # Current depth
    vl.log('current_depth')
    vl.log(current_depth)

    # Nothing to work on
    if wts is None or wts.shape[1] == 0:
        vl.log('Finished: Nothing to branch to.')
        return

    next_wts_list = []
    for wt_i in range(wts.shape[1]):
        wt = wts[:, wt_i]
        # Check if transversal is complete
        is_complete, completion, fts_response = verify_completion(H, wt, fts)
        if fts_response:
            vl.log('Finished: Maximum iterations hit.')
            return
        if is_complete:
            continue

        # Otherwise, branch to nodes that can contribute
        for node in branch_to_nodes(H, wt, completion):
            next_wt = continue_transveral(wt, node)
            next_wts_list.append(next_wt)

    # Nothing valid to continue on
    if len(next_wts_list) == 0:
        vl.log('Finished: All branches complete.')
        return

    next_wts_mtrx = sparse.hstack(next_wts_list)

    # Filter out minimal copies
    next_wts = cernikov_filter(next_wts_mtrx, fts.raw()) # Will iterate over columns

    work_on_transversals_breadth(H, next_wts, fts, current_depth + 1)

def work_on_transversals_depth(H, wt, fts, current_depth=0):
    """ Work on a particular transversal going depth first """
    # Current depth
    vl.log('current_depth')
    vl.log(current_depth)

    # Check if transversal is complete
    is_complete, completion, fts_response = verify_completion(H, wt, fts)
    if fts_response:
        vl.log('Finished: Maximum iterations hit.')
        return True # Finish everything
    if is_complete:
        return False # Branch is over, keep searching

    # Otherwise, branch to nodes that can contribute
    for node in branch_to_nodes(H, wt, completion):
        next_wt = continue_transveral(wt, node)
        if fts.minimal_present(next_wt):
            # A more minimal transversal is present, just continue on
            continue
        terminate = work_on_transversals_depth(H, next_wt, fts, current_depth + 1)
        if terminate:
            vl.log('Propagate finished.')
            return True # Finish everything

def is_complete_transversal(completion):
    """ Given a completion is it complete """
    # Completion is a row vector
    return completion.nnz == completion.shape[1] # No nonzero elements means completion

def transversal_completion(H, t):
    """ Tells me the edges that were hit already (and how many times) """
    return t.T * H

# def transversal_overlap(H, t):
#     """ Get the overlap between the edges of a transversal (completion) and the entire hypergraph """
#     return H * t.T

def continue_transveral(wt, node):
    """
    wt : working transversal
    node : node to be chosen
    """
    new_indices = np.append(wt.indices, [node])
    new_indptr = np.copy(wt.indptr)
    new_indptr[1] = len(new_indices)

    new_data = np_ones(len(wt.data) + 1)

    wt_new = sparse.csc_matrix((new_data, new_indices, new_indptr), shape=wt.shape)

    vl.log('new_t')
    vl.log(wt_new)

    return wt_new

def get_missing_edges(H, wt, completion):
    completion.data.fill(1) # unitize
    missing_edges = get_unit_completion(H.shape[1]) - completion # find what edges are missing
    return missing_edges

def branch_to_edges(H, wt, completion):
    missing_edges = get_missing_edges(H, wt, completion) # Obtain the missing edge sparse list
    edges = missing_edges.indices
    if len(edges) != 0:
        yield edges[0] # Even if it's not sorted, take a missing edge

# def edge_vector(edge, size):
#     return sparse.csr_matrix((np.ones(1), np.array([edge]), np.array([0, 1])),  shape=(1, size))

def branch_to_nodes(H, wt, completion):
    """
    decide which nodes to branch to next
    """
    next_edges = branch_to_edges(H, wt, completion)
    H = sparse.csc_matrix(H) # Ensures the matrix is csc format (should already be)
    for edge in next_edges:
        node_indices_to_contribute = H[:, edge].indices
#         edge_v = edge_vector(edge, H.shape[1])
#         nodes_to_contribute = transversal_overlap(H, edge_v)
#         node_indices_to_contribute = ptrflux(nodes_to_contribute.indptr)
        for i in node_indices_to_contribute:
            if not wt[i, 0] > 0: # not already part of working transversal
                vl.log('Branching to node')
                vl.log(i)
                yield i

vl = VerboseLog(False)
def perform_tests():
    # ABC_222_222
    row_sum, A, col_sum, contracted_A = get_contraction_elements(ABC_222_222)

    H = hyper_graph(A, 0)
    # plot_coo_matrix(H)
    H_bool = H.astype(bool)
    mediumH = sparse.csr_matrix(np.array([
                [1, 0, 1, 0, 0, 1, 0, 1],
                [0, 1, 0, 0, 1, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 1, 0],
                [1, 1, 1, 0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1, 0, 0, 1],
                [0, 1, 1, 1, 0, 1, 0, 1],
                [1, 1, 1, 0, 0, 0, 1, 1],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 1],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
            ]))
    smallH = sparse.csr_matrix(np.array([
                [1, 0, 1, 0, 1],
                [0, 1, 0, 0, 1],
                [1, 0, 1, 1, 0],
                [1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1],
                [0, 1, 1, 1, 0],
            ]))
    xsmallH = sparse.csr_matrix(np.array([
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 0],
            ]))
    # plot_coo_matrix(mediumH)
    print(hyper_graph(A, 0).toarray())
    fts = find_transversals(hyper_graph(A, 0), strat=HTStrat(search='depth', max_t=np.inf))
    print(fts.raw().toarray())

    # continue_transveral(get_null_transveral(6), 1)
    # PROFILE_MIXIN(find_transversals, hyper_graph(A, 11), 'depth')

if __name__ == '__main__':
    # pass
    PROFILE_MIXIN(perform_tests)