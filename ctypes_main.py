import time
import ctypes
import os
from ctypes import c_int

import numpy as np
ivec_1d = np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS")

c_utils = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "ctypes_utils.so"))


prime_N = 12345678
graph_V = 123456
graph_E = 1234567


# Define ctype interface to pass the numpy array
c_utils.count_primes.argtypes = [ctypes.c_int, ivec_1d]
c_utils.count_primes.restype = ctypes.c_int

time_before_prime = time.time()
non_prime = np.zeros(prime_N + 1, dtype=int)
prime_counter = c_utils.count_primes(prime_N, non_prime)
print("numer of primes", prime_counter)
print("time to count primes", time.time() - time_before_prime)


print("\n" + "=" * 60 + "\n")
data = np.load("data.npy")
X, Y, Z = data[0], data[1], data[2]


c_utils.dijkstra_heap.argtypes = [
    c_int, c_int, c_int, ivec_1d, ivec_1d, ivec_1d, ivec_1d, ivec_1d, ivec_1d, ivec_1d
]
c_utils.dijkstra_heap.restype = None

time_before_dijkstra = time.time()
for sta in range(1, 1 + 5):
    pos = np.zeros(graph_V + 9, dtype=int)
    e = np.zeros(graph_E + 9, dtype=int)
    vis = np.zeros(graph_V + 9, dtype=int)
    d = np.ones(graph_V + 9, dtype=int) * int(1e9)
    c_utils.dijkstra_heap(graph_V, graph_E, sta, X, Y, Z, pos, e, vis, d)
    tar = graph_V // sta
    print(f"distance from {sta} to {tar} = {d[tar]}")
print("time for dijkstra with (C++ STL) heap", time.time() - time_before_dijkstra)
print()

c_utils.dijkstra_segment.argtypes = [
    c_int, c_int, c_int, c_int, ivec_1d, ivec_1d, ivec_1d, ivec_1d, ivec_1d, ivec_1d, ivec_1d
]
c_utils.dijkstra_segment.restype = None

time_before_segment = time.time()
for sta in range(1, 1 + 5):
    pw2 = 1
    while pw2 <= graph_V:
        pw2 <<= 1
    c_utils.seg_init(graph_V, pw2)
    pos = np.zeros(graph_V + 9, dtype=int)
    e = np.zeros(graph_E + 9, dtype=int)
    vis = np.zeros(graph_V + 9, dtype=int)
    d = np.ones(graph_V + 9, dtype=int) * int(1e9)
    c_utils.dijkstra_segment(pw2, graph_V, graph_E, sta, X, Y, Z, pos, e, vis, d)
    tar = graph_V // sta
    print(f"distance from {sta} to {tar} = {d[tar]}")
print("time for dijkstra with segment tree", time.time() - time_before_segment)
print()