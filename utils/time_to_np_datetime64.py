
import time
import numpy as np
# from itertools import repeat

t0 = time.time()
for _ in range(10_000_000):
    tx = time.time_ns()
    tx_np = np.datetime64(tx,'ns')

tf = time.time() - t0
print(f"overall: {tf:.5f} s, looptime: {tf/10_000_000} s")    