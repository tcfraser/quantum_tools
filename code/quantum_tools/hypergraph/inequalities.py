from ..inflation import marginal_equality
import numpy as np

def get_b(pd, preinjectable_sets, orbit_contractor=None):
    """
    Converts from a probability distribution to a target for a particular inequality
    """
    b = marginal_equality.contexts_marginals(pd, preinjectable_sets) # The marginals over the preinjectable sets
    if orbit_contractor is not None:
        return orbit_contractor.dot(b) # The new contracted marginals into orbits
    else:
        return b

def pd_to_ineq_target(pd, preinjectable_sets, transversals, antecedent, hg_rows, multi=False, orbit_contractor=None):
    """
    pd: The Probability Distribution in question
    transversal(s): the transversal matrix that 
    transversals: The fts array
    """
    b = get_b(pd, preinjectable_sets, orbit_contractor)
    
    antecedent_value = b[antecedent] # The lhs of the inequality
    marginals_hg_space = b[hg_rows] # Same reduction made to get transveral

    num_targets = transversals.shape[1]
    if not multi and num_targets > 1:
        raise Exception("Not requesting multiple target values when multiple transversals sent.")
    targets = np.zeros(num_targets)
    for i in range(num_targets):
        indices = transversals[:, i].indices
        consequent_values = marginals_hg_space[indices] # The values of the particular consequents
        target = np.sum(consequent_values, axis=0) - antecedent_value # This value *should* be positive if no hardy paradox
        targets[i] = target
    if not multi:
        return targets[0]
    return targets