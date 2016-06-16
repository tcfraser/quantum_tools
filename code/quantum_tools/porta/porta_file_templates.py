# dim = \
# """DIM = {dim}
# """

# valid = \
# """VALID
# {valid}
# """

# lower_bounds = \
# """
# LOWER_BOUNDS
# {lower_bounds}
# """

# upper_bounds = \
# """UPPER_BOUNDS
# {upper_bounds}
# """

# elimination_order = \
# """
# ELIMINATION_ORDER
# {elimination_order}
# """

# inequalities_equalities_section = \
# """
# INEQUALITIES_SECTION
# {equalities_section}

# {inequalities_section}
# """

# valid_fields = [
#     'dim',
#     'valid',
#     'lower_bounds',
#     'upper_bounds',
#     'elimination_order',
#     'inequalities_equalities_section'
#     ]

# def add_EOF(str_file):
#     return str_file + """\nEND"""

# file_template = \
#     dim + \
#     valid + \
#     lower_bounds + \
#     upper_bounds + \
#     elimination_order + \
#     inequalities_equalities_section

# file_template = add_EOF(file_template)
