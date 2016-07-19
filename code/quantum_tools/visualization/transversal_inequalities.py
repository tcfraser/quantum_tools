from ..inflation import marginal_equality
from collections import defaultdict
from ..utilities import utils
from ..statistics import variable_sort
import numpy as np
# from itertools import product
# from operator import itemgetter

class Latex():
    """
    Views the output in latex format.
    """
    def __init__(self, *args):
        self.input = args
        self.raw = self._parse_input()
        
    def _parse_input(self):
        return '\n'.join(map("${0}$".format, self.input))
        
    def _repr_latex_(self):
        return self._parse_input()

def get_preinjectablesets_latex(rvc, preinjectable_sets):
    defl_map = marginal_equality.get_delf_map(preinjectable_sets)
    unflat = [get_preinjectableset_latex(rvc, pi, defl_map) for pi in preinjectable_sets]
    return np.array(list(utils.flatten(unflat)))

def _sitemgetter(iterable):
    return lambda x: [x[i] for i in iterable]

def get_preinjectableset_latex(rvc, preinjectable_set, defl_map):
    defl_preinjectable_latex = marginal_equality.deflate(preinjectable_set, defl_map)
    preinjectable_latex = []
    rv_names = list(utils.flatten(preinjectable_set))
    rv_names_sort = variable_sort.sort(rv_names)
    sort_map = dict((i, j) for j, i in enumerate(rv_names_sort))
    sort_getter = [_sitemgetter(tuple(sort_map[rv_name] for rv_name in i)) for i in preinjectable_set]
    rv_zip = list(zip(defl_preinjectable_latex, sort_getter))
    sub_rvc = rvc.sub(rv_names)
    for outcome in sub_rvc.outcome_space:
        definite_labels = [definite_outcome_latex(i, sort_get(outcome)) for i, sort_get in rv_zip]
        preinjectable_latex.append(''.join(map(probabilize, definite_labels)))
    return preinjectable_latex

def definite_outcome_latex(rv_names, outcomes):
    joint_outcome_latex = ''.join("{0}_{1}".format(rv_name, outcome) for outcome, rv_name in zip(outcomes, rv_names))
    return joint_outcome_latex

def probabilize(event):
    return "P({0})".format(event)

def get_equation_latex(terms, lhs, relation, rhs):
    hs_latex = [None, None] # LHS and RHS goes here
    
    for i, side in enumerate([lhs, rhs]):
        coeff_map = defaultdict(int)
        
        for q in side:
            coeff_map[terms[q]] += 1
        
        latexed_terms = []
        for term, coeff in coeff_map.items():
            if coeff == 1:
                latexed_terms.append(term)
            else:
                latexed_terms.append(str(coeff) + term)
        latexed_terms = sorted(latexed_terms)
        hs_latex[i] = ' + '.join(latexed_terms)
           
    result = "{lhs} {rel} {rhs}".format(lhs=hs_latex[0], rel=relation, rhs=hs_latex[1])
    return result

def transversal_inequality(ant, cons, hg_rows, b):
    """
    ant: The index of the antecedent
    cons: The transversal indices
    hg_rows: How the rows were contracted to get the hypergraph
    b: The latexed terms (see get_preinjectablesets_latex)
    """
    return get_equation_latex(b, [ant], '\leq', hg_rows[cons])

def get_duplication_map(b):
    dup_map = defaultdict(list)
    for key, val in enumerate(b):
        dup_map[val].append(key)
    return dup_map