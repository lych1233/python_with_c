import time

import numpy as np
import cython_utils as c_utils


prime_N = 12345678
graph_V = 123456
graph_E = 1234567


time_before_prime = time.time()
non_prime = np.zeros(prime_N + 1, dtype=int)
prime_counter = c_utils.count_primes(prime_N, non_prime)
print("number of primes", prime_counter)
print("time to count primes", time.time() - time_before_prime)


print("\n" + "=" * 60 + "\n")
data = np.load("data.npy")
X, Y, Z = data[0], data[1], data[2]

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

time_before_segment = time.time()
pw2 = 1
while pw2 <= graph_V:
    pw2 <<= 1
dijktra_segment_solver = c_utils.DijkSeg(graph_V, pw2)
for sta in range(1, 1 + 5):
    dijktra_segment_solver.__init__(graph_V, pw2)
    pos = np.zeros(graph_V + 9, dtype=int)
    e = np.zeros(graph_E + 9, dtype=int)
    vis = np.zeros(graph_V + 9, dtype=int)
    d = np.ones(graph_V + 9, dtype=int) * int(1e9)
    dijktra_segment_solver.dijkstra_segment(pw2, graph_V, graph_E, sta, X, Y, Z, pos, e, vis, d)
    tar = graph_V // sta
    print(f"distance from {sta} to {tar} = {d[tar]}")
print("time for dijkstra with segment tree", time.time() - time_before_segment)
print()
