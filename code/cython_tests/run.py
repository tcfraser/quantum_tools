import pyximport; pyximport.install()
import helloworld
import timeit

import primes; print(primes.primes(1000))
print(timeit.timeit('import primes; primes.primes(1000)', number=100))
print(timeit.timeit('import primes_regular; primes_regular.primes(1000)', number=100))