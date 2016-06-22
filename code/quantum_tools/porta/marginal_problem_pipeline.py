from ..porta.porta_tokenizer import Term, Terms, Relation, PortaFile
from ..porta import porta_tokenizer
from fractions import Fraction
import numpy as np
from scipy import sparse, io
import math
import os
import shutil
import time
import subprocess

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPE_LINE_DIRECTORY = os.path.join(_THIS_DIR, 'pipeline_directory')
_BIN_FILES = os.path.join(_THIS_DIR, 'src', 'porta-1.4.1', 'gnu-make', 'bin')
_PORTA_EXE = os.path.join(_BIN_FILES, 'xporta.exe')

def _build_vm(start, stop):
    # Format for variable names
    return np.array(["x{n}".format(n=i+1) for i in range(start, stop)], dtype=str)

# Basic term elements
__PLUS_SIGN, __ONE_FRACTION = '+', Fraction(1)
__ZERO_TERM = Term(__PLUS_SIGN, Fraction(0), '')
__ZERO_TERMS = Terms([__ZERO_TERM])
# def _termify(sign_value_name):
#     f_sign, f_value, variable_name = sign_value_name[0], sign_value_name[1], sign_value_name[2]
#     sign = '+' if f_sign == '1.0' else '-'
#     fraction = Fraction(f_value)
#     term = Term(sign, fraction, variable_name)
#     print(term)
#     return [term]
# _termify = np.vectorize(_termify)

def convert_to_porta_file(A):
    # Ensure csr format for indexing efficiency
    if A.getformat() is not 'csr':
        A = A.tocsr()
    # Setup row and col variable maps
    A_nrow, A_ncol = A.shape
    indptr, indices, data = A.indptr, A.indices, A.data
    row_vm = _build_vm(     0, A_nrow +      0)
    col_vm = _build_vm(A_nrow, A_nrow + A_ncol)
    # t_row_vm = _termify(row_vm)
    # t_col_vm = _termify(col_vm)

    # Portafile object
    pf_obj = {}

    # Dimension
    pf_obj['dim'] = A_nrow + A_ncol

    # Equalities
    equalities = []
    for row_i in range(A_nrow):
        col_indices = indices[indptr[row_i]:indptr[row_i+1]]
        row_data = data[indptr[row_i]:indptr[row_i+1]]
        sum_terms = []
        for j in range(len(row_data)):
            data_sign = math.copysign(1,row_data[j])
            data_abs = abs(row_data[j])
            variable_name = col_vm[col_indices[j]]
            data_sign = '+' if data_sign >= 0 else '-'
            data_abs = Fraction(data_abs)
            sum_terms.append(Term(data_sign, data_abs, variable_name))
        rhs_term = Term(__PLUS_SIGN, __ONE_FRACTION, row_vm[row_i])
        rel = Relation(Terms(sum_terms), '==', Terms([rhs_term]))
        equalities.append(rel)
    pf_obj['equalities_section'] = equalities

    # Inequalities
    inequalities = []
    for variable_name in col_vm:
        term = Term(__PLUS_SIGN, __ONE_FRACTION, variable_name)
        rel = Relation(Terms([term]), '>=', __ZERO_TERMS)
        inequalities.append(rel)
    pf_obj['inequalities_section'] = inequalities

    # Elimination Order
    elim_order = [0] * A_nrow + list(range(1, A_ncol+1))
    pf_obj['elimination_order'] = elim_order

    pf = PortaFile.fromObj(pf_obj, template='input')
    return pf, row_vm, col_vm

def save_sparse_mtrx_to_file():
    pass

def perform_pipeline(name, mtrx, optional_mtrxs=None):
    working_dir = os.path.join(_PIPE_LINE_DIRECTORY, name)
    print("Using directory {0}.".format(working_dir))
    if os.path.exists(working_dir):
        print("Removing previous directory {0}.".format(working_dir))
        shutil.rmtree(working_dir)
    os.makedirs(working_dir)
    if optional_mtrxs is not None:
        for name, optional_mtrx in optional_mtrxs.items():
            save_slot = os.path.join(working_dir, name + '.mtx')
            print("Saving optional matrix {0}.".format(save_slot))
            io.mmwrite(save_slot, optional_mtrx)
    setup_file_name = os.path.join(working_dir, 'setup.ieq')
    fmel_log_file_name = os.path.join(working_dir, 'porta_fmel.log')
    print("Making setup file {0}.".format(setup_file_name))
    pf, row_vm, col_vm = convert_to_porta_file(mtrx)
    pf.toFile(setup_file_name)
    print("Calling fourier motzkin elimination:")
    subprocess_call = [_PORTA_EXE, '-F', setup_file_name]
    print(" ".join(subprocess_call))
    p = subprocess.Popen(
        args=subprocess_call,
        close_fds=True,
        bufsize=-1,
        # shell=True, # Nope.
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
    )
    output, error = p.communicate()
    print("Writing log file {0}.".format(fmel_log_file_name))
    if not error:
        with open(fmel_log_file_name, 'w+') as _file:
            _file.write(output)
    else:
        raise Exception("return code for porta was {0}".format(error))
    file_to_clean = os.path.join(working_dir, 'setup.ieq.ieq')
    print("Cleaning file {0}.".format(file_to_clean))
    cleaned_file = PortaFile.clean(file_to_clean)
    return cleaned_file