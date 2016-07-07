from collections import namedtuple

SymbolicContext = namedtuple('SymbolicContext', ['preinjectable_sets', 'outcomes'])

ABC_444_444 = SymbolicContext(
    preinjectable_sets=[
        [['A1', 'B1', 'C1'], ['A4', 'B4', 'C4']],
        [['A1', 'B2', 'C3'], ['A4', 'B3', 'C2']],
        [['A2', 'B3', 'C1'], ['A3', 'B2', 'C4']],
        [['A2', 'B4', 'C3'], ['A3', 'B1', 'C2']],
        [['A1'], ['B3'], ['C4']],
        [['A1'], ['B4'], ['C2']],
        [['A2'], ['B1'], ['C4']],
        [['A2'], ['B2'], ['C2']],
        [['A3'], ['B3'], ['C3']],
        [['A3'], ['B4'], ['C1']],
        [['A4'], ['B1'], ['C3']],
        [['A4'], ['B2'], ['C1']],
    ],
    outcomes=[4]*(4 + 4 + 4),
)
ABC_222_222 = SymbolicContext(
    preinjectable_sets=[
        [['A2'], ['B2'], ['C2']],
        [['B2'], ['A2',   'C1']],
        [['C2'], ['A1',   'B2']],
        [['A2'], ['B1',   'C2']],
        [['A1',   'B1',   'C1']],
    ],
    outcomes=[2]*(2 + 2 + 2),
)
ABC_222_444 = SymbolicContext(
    preinjectable_sets=[
        [['A2'], ['B2'], ['C2']],
        [['B2'], ['A2',   'C1']],
        [['C2'], ['A1',   'B2']],
        [['A2'], ['B1',   'C2']],
        [['A1',   'B1',   'C1']],
    ],
    outcomes=[4]*(2 + 2 + 2),
)
ABC_224_444 = SymbolicContext(
    preinjectable_sets=[
        [['C1'], ['A2', 'B2', 'C4']],
        [['C2'], ['A1', 'B2', 'C3']],
        [['C3'], ['A2', 'B1', 'C2']],
        [['C4'], ['A1', 'B1', 'C1']],
    ],
    outcomes=[4]*(2 + 2 + 4),
)

ABXY_2222_2222 = SymbolicContext(
    preinjectable_sets= [
        [['X1', 'Y1'], ['A2', 'B2', 'X2', 'Y2']],
        [['X1', 'Y2'], ['A2', 'B1', 'X2', 'Y1']],
        [['X2', 'Y1'], ['A1', 'B2', 'X1', 'Y2']],
        [['X2', 'Y2'], ['A1', 'B1', 'X1', 'Y1']],
    ],
    outcomes=[2]*(2+2+2+2),
)