import time
import heapq

import numpy as np


prime_N = 12345678
graph_V = 123456
graph_E = 1234567


def count_primes(n):
    for i in range(2, n):
        for j in range(2, n // i):
            non_prime[i * j] = 1
    prime_counter = 0
    for i in range(2, n):
        if not non_prime[i]:
            prime_counter += 1
    return prime_counter

time_before_prime = time.time()
non_prime = np.zeros(prime_N + 1, dtype=int)
prime_counter = count_primes(prime_N)
print("numer of primes", prime_counter)
print("time to count primes", time.time() - time_before_prime)


print("\n" + "=" * 60 + "\n")
data = np.load("data.npy")
X, Y, Z = data[0], data[1], data[2]


def dijkstra_heap(n, m, sta, x, y, z):
    for i in range(1, m + 1):
        pos[x[i]] += 1
    for i in range(1, n + 1):
        pos[i] += pos[i - 1]
    for i in range(1, m + 1):
        e[pos[x[i]]] = i
        pos[x[i]] -= 1
    for i in range(1, n + 1):
        pos[i] += 1
    pos[n + 1] = m + 1

    heapq.heapify(Q)
    heapq.heappush(Q, (0, sta))
    d[sta] = 0
    while len(Q):
        _, u = heapq.heappop(Q)
        if vis[u]:
            continue
        vis[u] = 1
        for i in range(pos[u], pos[u + 1]):
            e_i = e[i]
            v, w = y[e_i], z[e_i]
            if d[u] + w < d[v]:
                d[v] = d[u] + w
                heapq.heappush(Q, (d[v], v))

time_before_dijkstra = time.time()
for sta in range(1, 1 + 5):
    pos = np.zeros(graph_V + 9, dtype=int)
    e = np.zeros(graph_E + 9, dtype=int)
    vis = np.zeros(graph_V + 9, dtype=bool)
    d = np.ones(graph_V + 9, dtype=int) * int(1e9)
    Q = []
    dijkstra_heap(graph_V, graph_E, sta, X, Y, Z)
    tar = graph_V // sta
    print(f"distance from {sta} to {tar} = {d[tar]}")
print("time for dijkstra with heap", time.time() - time_before_dijkstra)
print()


def seg_init(n, pw2):
    for i in range(1, 1 + n):
        seg_pos[i | pw2] = i
    for i in range((n | pw2) >> 1, 1, -1):
        if seg_min[i << 1] < seg_min[i << 1 | 1]:
            seg_pos[i] = seg_pos[i << 1]
        else:
            seg_pos[i] = seg_pos[i << 1| 1]

def seg_upd(x, y):
    k = x | pw2
    seg_min[k] = y
    while k > 1:
        k >>= 1
        l = k << 1 | (seg_min[k << 1 | 1] < seg_min[k << 1])
        seg_min[k], seg_pos[k] = seg_min[l], seg_pos[l]

def dijkstra_segment(n, m, sta, x, y, z):
    for i in range(1, m + 1):
        pos[x[i]] += 1
    for i in range(1, n + 1):
        pos[i] += pos[i - 1]
    for i in range(1, m + 1):
        e[pos[x[i]]] = i
        pos[x[i]] -= 1
    for i in range(1, n + 1):
        pos[i] += 1
    pos[n + 1] = m + 1

    seg_upd(sta, 0)
    d[sta] = 0
    while seg_min[1] < int(1e9):
        u = seg_pos[1]
        if vis[u]:
            continue
        vis[u] = 1
        seg_upd(u, int(1e9))
        for i in range(pos[u], pos[u + 1]):
            e_i = e[i]
            v, w = y[e_i], z[e_i]
            if d[u] + w < d[v]:
                d[v] = d[u] + w
                seg_upd(v, d[v])

time_before_segment = time.time()
for sta in range(1, 1 + 5):
    seg_min = np.ones(graph_V * 4 + 9, dtype=int) * int(1e9)
    seg_pos = np.zeros(graph_V * 4 + 9, dtype=int)
    pw2 = 1
    while pw2 <= graph_V:
        pw2 <<= 1
    seg_init(graph_V, pw2)
    pos = np.zeros(graph_V + 9, dtype=int)
    e = np.zeros(graph_E + 9, dtype=int)
    vis = np.zeros(graph_V + 9, dtype=bool)
    d = np.ones(graph_V + 9, dtype=int) * int(1e9)
    dijkstra_segment(graph_V, graph_E, sta, X, Y, Z)
    tar = graph_V // sta
    print(f"distance from {sta} to {tar} = {d[tar]}")
print("time for dijkstra with segment tree", time.time() - time_before_segment)
print()
