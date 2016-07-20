from ..examples.symbolic_contexts import *
from ..symmetries.workspace import *
from ..visualization.sparse_vis import plot_matrix
from ..config import *
from scipy import sparse
import numpy as np
from functools import reduce
from operator import mul
import inspect

def slog(*args):
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe)[2]
    print('[{name}:{lineno}]:'.format(name=calframe[3], lineno=calframe[2]), *args)

def nzcfr(M, row):
    """ Non zero columns for a given row of a csr matrix """
    return M.indices[M.indptr[row]:M.indptr[row+1]]

def ptrflux(indptr):
    """ Compute the indices flux, determining which elements contain data """
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

def hyper_graph_contraction(A, ant, remove=None):
    """ Prefers csr format. Find contraction for hypergraph. """
    A_csr = sparse.csr_matrix(A)
    # Take only columns that are extendable versions of the antescedant (ant)
    ext_ant = nzcfr(A_csr, ant)
    ext_A = A_csr[:, ext_ant]
    # Take only rows that would contribute
    row_entry_flux = ext_A.indptr[1:] - ext_A.indptr[:-1] # Finds where the row pointers change
    if remove is not None:
        for i in remove:
            row_entry_flux[i] = 0 # Make antecedant zero so it doesn't get selected (as well as duplicates)

    H_rows = np.nonzero(row_entry_flux)[0] # The rows that should be used in the hypergraph
    H_cols = ext_ant # The cols that should be used in the hypergraph
    return H_rows, H_cols

def hyper_graph(A, ant, remove=None):
    if isinstance(A, dict):
        A_csr = A['csr']
        A_csc = A['csc']
    else:
        A_csr = sparse.csr_matrix(A)
        A_csc = sparse.csc_matrix(A)
    
    if remove is None:
        remove = [ant]   
    H_rows, H_cols = hyper_graph_contraction(A_csr, ant, remove=remove)
    
    H = A_csc[:, H_cols][H_rows, :]
    
    return H_rows, H, H_cols

# ======================
# === Misc Utilities ===
# ======================
def which_nodes_essential(hg):
    """
    Determines which nodes are absolutely needed for any transversal.
    Corresponds to an edge that only has one node hitting it.
    """
    hg = sparse.csc_matrix(hg, copy=True)
    hg.data.fill(1)
    where_edges = np.array(hg.sum(axis=0) == 1).flatten()
    reduced = hg[:, where_edges]
    return reduced.indices

def which_edges_greedy(hg):
    """
    Which edges hit every node of the hypergraph.
    """
    return np.where(hg.sum(axis=1) == hg.shape[1])

def transversals_exist(hg):
    """
    Determines if there are any transversals at all.
    Corresponds to being able to hit every edge with at least some node.
    """
    return not np.any(hg.sum(axis=0) == 0)

EXP_DTYPE = 'int16' # Can't be bool. Scipy sparse arrays don't support boolean matrices well

class FoundTransversals():

    def __init__(self, log=False):
        self._raw = None
        self._raw_sum = None
        self.__log = log

    @staticmethod
    def from_existing(raw, log=False):
        fts = FoundTransversals(log)
        if raw is not None:
            fts._raw = raw
            fts._raw_sum = np.array(raw.sum(axis=0)).flatten()
        return fts
        
    def __len__(self):
        if self._raw is None:
            return 0
        else:
            return self._raw.shape[1]
        
    def log(self, *args):
        if self.__log:
            slog(*args)

    def __getitem__(self, slice):
        if self._raw is None:
            return None
        else:
            return self._raw[:, slice]

    def __iter__(self):
        if self._raw is not None:
            for i in range(self._raw.shape[1]):
                yield self._raw[:, i]

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
        self.log('Added ft', len(self))

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
    return wts

#============================
#==== Pooled Resources ======
#============================
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

class TransversalStrat():

    def __init__(self, **kwargs):
        config = kwargs
        self.search_type = config.get('search_type')
        self.find_up_to = config.get('find_up_to', np.inf)
        self.node_brancher = config.get('node_brancher')
        self.breadth_cap = config.get('breadth_cap', np.inf)
        self.__filter_out = config.get('filter_out', None)
        self.starting_transversal = config.get('starting_transversal', None)
        if self.node_brancher is not None:
            self.max_node_branch = self.node_brancher.get('max', np.inf)
            self.ignore_nodes = self.node_brancher.get('ignore', None)
        else:
            self.max_node_branch = np.inf
            self.ignore_nodes = None
            
    def filter_out(self, wt):
        if self.__filter_out is None:
            return False # Don't filter out
        else:
            return self.__filter_out(wt)
        
    def get_starting_transversal(self, num_nodes):
        if self.starting_transversal is None:
            return get_null_transveral(num_nodes)
        else:
            formatted_st = sparse.csc_matrix(self.starting_transversal, dtype=EXP_DTYPE, copy=True)
            assert(formatted_st.shape == (num_nodes, 1)), "Starting transversal has invalid shape {}, needs to be {}".format(formatted_st.shape, (num_nodes, 1))
            return formatted_st
        
    # def precache(self, trans):
    #     if self.ignore_nodes is not None:
    #         ref = np.asarray(self.ignore_nodes)
    #         self.ignore_nodes = np.zeros(trans.num_nodes, dtype=bool)
    #         for i in np.nditer(ref):
    #             self.ignore_nodes[i] = True      

class HGT():

    STOP = 'STOP'
    CONTINUE = 'CONTINUE'

    @staticmethod
    def completion(H, t):
        """
        Determines which edges have been included in a given transversal t
        Note: Returns a sparse row vector
        """
        return t.T * H

    @staticmethod
    def is_complete(completion):
        """
        Given a completion is it complete?
        Note: Completion is a row vector
        """
        return completion.nnz == completion.shape[1] # No nonzero elements means completion

    @staticmethod
    def append_node(wt, node):
        """
        wt : working transversal
        node : node to be chosen (assumed not already present)
        """
        new_indices = np.append(wt.indices, [node])
        new_indptr = np.copy(wt.indptr)
        new_indptr[1] = len(new_indices)

        new_data = np_ones(len(wt.data) + 1)

        wt_new = sparse.csc_matrix((new_data, new_indices, new_indptr), shape=wt.shape)

        return wt_new
    
    @staticmethod
    def remove_ith_node(wt, i):
        """
        wt: transversal
        node: node to be removed (assumed present)
        """
        new_indices = np.delete(wt.indices, i)
        new_indptr = np.copy(wt.indptr)
        new_indptr[1] = len(new_indices)

        new_data = np_ones(len(wt.data) - 1)

        wt_new = sparse.csc_matrix((new_data, new_indices, new_indptr), shape=wt.shape)

        return wt_new

    @staticmethod
    def get_missing_edges(completion):
        completion.data.fill(1) # unitize
        missing_edges_sparse = get_unit_completion(completion.shape[1]) - completion # find what edges are
        missing_edges = missing_edges_sparse # Keep in sparse format
        return missing_edges
    
    @staticmethod
    def is_minimal(H, t):
        for i in range(t.nnz):
            sub_t = HGT.remove_ith_node(t, i)
            sub_completion = HGT.completion(H, sub_t)
            sub_is_complete = HGT.is_complete(sub_completion)
            if sub_is_complete:
                return False
        return True

    @staticmethod
    def _make_minimal_single(H, t):
        i = 0
        while i < t.nnz:
            sub_t = HGT.remove_ith_node(t, i)
            sub_completion = HGT.completion(H, sub_t)
            sub_is_complete = HGT.is_complete(sub_completion)
            if sub_is_complete:
                t = sub_t
            else:
                i += 1
        return t

    @staticmethod
    def make_minimal(H, ts):
        ts = sparse.csc_matrix(ts)
        mts = []
        for i in range(ts.shape[1]):
            t = ts[:, i]
            minimal_t = HGT._make_minimal_single(H, t)
            mts.append(minimal_t)
        mts = sparse.hstack(mts, format='csc')
        return mts

    @staticmethod
    def sub_sample_ts(ts, n):
        choices = np.random.choice(np.arange(ts.shape[1]), n)
        sample = ts[:, choices]
        return sample

class HyperGraphTransverser():

    def __init__(self, H, strat, fts, log=False):
        self.H = sparse.csc_matrix(H)
        self.strat = strat if strat is not None else TransversalStrat()
        self.__log = log
        self.fts = fts if fts is not None else FoundTransversals()
        self.num_nodes = self.H.shape[0]
        self.node_weights = np.array(self.H.sum(axis=1), dtype='int64').flatten()
        self.num_edges = self.H.shape[1]
        self.edge_weights = np.array(self.H.sum(axis=0), dtype='int64').flatten()
        # self.strat.precache(self)
        
        self.transverse(self.strat.get_starting_transversal(self.num_nodes))

    def log(self, *args):
        """ Log info regarding the transversal algorithm. """
        if self.__log:
            slog(*args)

    def verify_completion(self, wt):
        """
        Determines if the working transversal is complete and if so,
        store it in the list of found Transversals

        Returns: is_complete, completion
        """
        completion = HGT.completion(self.H, wt)
        self.log('completion')
        self.log(completion)
        self.log('is_completion:', HGT.is_complete(completion))
        if HGT.is_complete(completion):
            self.fts.update(wt) # Update the list of transversals with the new found one
            return True, None # If complete, one shouldn't be doing anything with the completion
        return False, completion

    def transverse(self, wts, current_depth=0):
        if self.strat.search_type == 'breadth':
            return self.transverse_breadth(wts, current_depth)
        elif self.strat.search_type == 'depth':
            return self.transverse_depth(wts, current_depth)
        else:
            raise ValueError("Invalid strat.search_type: {0}".format(self.strat.search_type))

    def transverse_breadth(self, wts, current_depth=0):
        """ Work on a particular transversal going breadth first """
        self.log('current_depth', current_depth)

        # Nothing to work on
        if wts is None or wts.shape[1] == 0:
            self.log('Finished: Nothing to branch to.')
            return HGT.CONTINUE

        next_wts_list = []
        for wt_i in range(wts.shape[1]):
            wt = wts[:, wt_i]
            # Check if transversal is complete
            if self.strat.filter_out(wt):
                self.log('Transversal filtered out.')
                continue
            is_complete, completion = self.verify_completion(wt)
            if len(self.fts) >= self.strat.find_up_to: # Stop if obtained enough transversals
                self.log('Finished: Maximum transversals found.')
                return HGT.STOP
            if is_complete: # If this particular transversal is complete, it would have been updated
                continue

            # Otherwise, branch to nodes that can contribute
            for node in self.branch_to_nodes(wt, completion):
                next_wt = HGT.append_node(wt, node)
                next_wts_list.append(next_wt)

        # Nothing valid to continue on
        if len(next_wts_list) == 0:
            self.log('Finished: All branches complete.')
            return HGT.CONTINUE

        next_wts = sparse.hstack(next_wts_list)

        # Memory cap
        # print('debug', self.strat.breadth_cap, next_wts.shape[1])
        if self.strat.breadth_cap < next_wts.shape[1]:
            next_wts = next_wts[:, 0:self.strat.breadth_cap] 
            self.log("Trimming breadth transversals down to {0}".format(next_wts.shape[1]))
        # Filter out minimal copies
        next_wts = cernikov_filter(next_wts, self.fts.raw()) # Will iterate over columns
        
        return self.transverse(next_wts, current_depth + 1)

    def transverse_depth(self, wt, current_depth=0):
        """ Work on a particular transversal going depth first """
        self.log('current_depth', current_depth)

        # Check if transversal is complete
        is_complete, completion = self.verify_completion(wt)
        if self.strat.filter_out(wt):
            self.log('Transversal filtered out.')
            return HGT.CONTINUE # Branch is over, filtered out by custom filter
        if len(self.fts) >= self.strat.find_up_to:
            self.log('Finished: Maximum transversals found.')
            return HGT.STOP # Finish everything
        if is_complete:
            return HGT.CONTINUE # Branch is over, keep searching

        # Otherwise, branch to nodes that can contribute
        for node in self.branch_to_nodes(wt, completion):
            next_wt = HGT.append_node(wt, node)
            if self.fts.minimal_present(next_wt):
                # A more minimal transversal is present, just continue on
                continue
            ret = self.transverse(next_wt, current_depth + 1)
            # prevents branching
            if ret == HGT.STOP:
                self.log('Propagate finished.')
                return HGT.STOP # Finish everything

    def branch_to_nodes(self, wt, completion):
        """
        Decide which nodes to branch to next
        """
        missing_edges = HGT.get_missing_edges(completion) # Obtain the missing edge sparse list

        nb = self.strat.node_brancher
        
        # Determine if there is a maximum count
        count_max = min(self.strat.max_node_branch, self.num_nodes)
        ignore_nodes = self.strat.ignore_nodes
        
        if nb is None or not 'name' in nb: # Default
            # Gets nodes that contribute to missing edge
            edge = missing_edges.indices[0] # Grab any next edge
            node_indices = self.H[:, edge].indices
        elif nb['name'] == 'greedy' or nb['name'] == 'long':
            # Gets the nodes that overlap the most(least) with what's missing
            overlap = self.H.dot(missing_edges.T)
            # k = min(count_max + wt.nnz, overlap.nnz)
            k = min(count_max, overlap.nnz)
            if k >= self.num_nodes or k == overlap.nnz:
                if nb['name'] == 'greedy':
                    alg_slice = np.argsort(overlap.data)[::-1]
                else: # long
                    alg_slice = np.argsort(overlap.data)
            else: # Else be smart, don't perform O(nlogn) operations, perform O(k) operations
                if nb['name'] == 'greedy':
                    alg_slice = np.argpartition(overlap.data, -k)[-k:]
                else: #long
                    alg_slice = np.argpartition(overlap.data, k)[:k]
            node_indices = overlap.indices[alg_slice]
        elif nb['name'] == 'random':
            # Gets nodes that contribute to random missing edge
            edge = np.random.choice(missing_edges.indices) # Grab any next edge
            node_indices = self.H[:, edge].indices
        elif nb['name'] == 'diverse':
            # Diversify the kinds of transversals that have been found
            if wt.nnz == 0: # Just starting out
                node_indices = np.arange(self.num_nodes) # Branch to everything
            else: # Otherwise be greedy up to one
                # edge = missing_edges.indices[0] # Grab any next edge
                # node_indices = [self.H[:, edge].indices[0]]
                # overlap = self.H.dot(missing_edges.T)
                # node_indices = [overlap.indices[np.argmax(overlap.data)]]
                scaled_overlap = overlap.data / (self.node_weights[overlap.indices]**2)
                node_indices = overlap.indices[np.where(np.max(scaled_overlap) == scaled_overlap)]
        else:
            raise ValueError("Invalid strat.node_brancher: {0}".format(self.strat.node_brancher))
        
        if nb is not None and bool(nb.get('shuffle', False)):
            np.random.shuffle(node_indices)
        
        count = 0
        for i in node_indices:
            if count >= count_max:
                break
            if not wt[i, 0] > 0: # not already part of working transversal
                if ignore_nodes is None or not ignore_nodes[i]:
                    self.log('Branching to node:', i)
                    count += 1
                    yield i

#================================================
#================ Main Method ===================
#================================================
def find_transversals(H, strat=None, log_wt=False, log_ft=False, fts=None):
    """ Header to call to begin everything """
    if fts is None:
        fts = FoundTransversals(log_ft)
    else:
        fts = FoundTransversals.from_existing(fts, log_ft)
    hgt = HyperGraphTransverser(H, strat, fts, log_wt)
    return hgt.fts.raw()
#================================================
#================================================
#================================================

def perform_tests():
    # ABC_222_222
    row_sum, A, col_sum, contracted_A = get_contraction_elements(ABC_222_222)

    H = hyper_graph(A, 0)
    # plot_matrix(H)
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
    # plot_matrix(mediumH)
    # print(hyper_graph(A, 0))
    strat = TransversalStrat(
        search_type='depth',
        find_up_to=10,
        node_brancher={
            'name': 'greedy',
            'max': 5,
        }
    )
    fts = find_transversals(H, strat=strat, log=False)
    fts_array = fts.toarray()
    print(fts_array)
    print(fts_array.shape)

    # continue_transveral(get_null_transveral(6), 1)
    # PROFILE_MIXIN(find_transversals, hyper_graph(A, 11), 'depth')

if __name__ == '__main__':
    # pass
    # PROFILE_MIXIN(perform_tests)
    perform_tests()