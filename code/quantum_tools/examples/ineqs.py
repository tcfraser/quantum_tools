def HLP3(PD):
    H = PD.H
    I = PD.I
    result = H(['A', 'B']) - I(['A', 'B', 'C']) + I(['A', 'B']) + I(['B', 'C']) + I(['C', 'A'])
    return result

def HLP2(PD):
    H = PD.H
    I = PD.I
    result = H('A') + H('B') + H('C') - 2 * (I(['A', 'B', 'C']) + I(['A', 'B']) + I(['B', 'C']) + I(['C', 'A']))
    return result

def HLP1(PD):
    H = PD.H
    I = PD.I
    result = H('A') - I(['A', 'B']) - I(['A', 'C'])
    return result