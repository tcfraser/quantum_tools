def primes(kmax):
    k = 0
    n = 2
    p = [0]*1000
    result = []
    if kmax > 1000:
        kmax = 1000
    while k < kmax:
        i = 0
        while i < k and n % p[i] != 0:
            i = i + 1
        if i == k:
            p[k] = n
            k = k + 1
            result.append(n)
        n = n + 1
    return result
