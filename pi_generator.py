from numpy.random import uniform as npru
import math


def rand_xy():
    x = npru(low=0.0, high=1.0)
    y = npru(low=0.0, high=1.0)
    return x, y


total_ct = 0
lt_one_ct = 0
num_iters = 1_000_000

for i in range(num_iters):
    total_ct += 1
    x, y = rand_xy()
    if (x * x + y * y) <= 1:
        lt_one_ct += 1

    if i % 100_000 == 0:
        pi_est = 4 * (lt_one_ct / total_ct)
        print(
            f"Iter {i} -- Pi estimate: {pi_est:.8f}, Error: {math.pi-pi_est:.8f}",
            flush=True,
        )

print(f"Final Pi estimate: {pi_est:.8f}, Error: {math.pi-pi_est:.8f}")
