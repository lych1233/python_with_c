# distutils: language = c++
from libcpp.pair cimport pair
from libcpp.queue cimport priority_queue
from libcpp.vector cimport vector

cimport numpy as np
np.import_array()


def count_primes(int n, int [:] non_prime):
    cdef int i, j
    for i in range(2, n):
        for j in range(2, n // i):
            non_prime[i * j] = 1
    cdef int prime_counter = 0
    for i in range(2, n):
        if not non_prime[i]:
            prime_counter += 1
    return prime_counter

ctypedef pair[int, int] ipair
cdef priority_queue[ipair] Q

def dijkstra_heap(int n, int m, int sta, int [:] x, int [:] y, int [:] z, int [:] pos, int [:] e, int [:] vis, int [:] d):
    cdef int i
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

    while not Q.empty():
        Q.pop()
    Q.push(ipair(0, sta))
    d[sta] = 0
    cdef int u, v, w, e_i
    cdef ipair o
    while not Q.empty():
        o = Q.top()
        u = o.second
        Q.pop()
        if vis[u]:
            continue
        vis[u] = 1
        for i in range(pos[u], pos[u + 1]):
            e_i = e[i]
            v, w = y[e_i], z[e_i]
            if d[u] + w < d[v]:
                d[v] = d[u] + w
                Q.push(ipair(-d[v], v))

cdef class DijkSeg(object):
    cdef int [40000000] seg_min
    cdef int [40000000] seg_pos
    cdef int inf
    def __init__(self, int n, int pw2):
        self.inf = int(1e9)
        cdef int i
        for i in range(4 * n):
            self.seg_min[i] = self.inf
            self.seg_pos[i] = 0
        for i in range(1, 1 + n):
            self.seg_pos[i | pw2] = i
        for i in range((n | pw2) >> 1, 1, -1):
            if self.seg_min[i << 1] < self.seg_min[i << 1 | 1]:
                self.seg_pos[i] = self.seg_pos[i << 1]
            else:
                self.seg_pos[i] = self.seg_pos[i << 1| 1]

    cdef seg_upd(self, int pw2, int x, int y):
        cdef int k = x | pw2
        self.seg_min[k] = y
        cdef int l
        while k > 1:
            k >>= 1
            l = k << 1 | (self.seg_min[k << 1 | 1] < self.seg_min[k << 1])
            self.seg_min[k], self.seg_pos[k] = self.seg_min[l], self.seg_pos[l]

    def dijkstra_segment(self, int pw2, int n, int m, const int sta, int [:] x, int [:] y, int [:] z, int [:] pos, int [:] e, int [:] vis, int [:] d):
        cdef int i
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

        self.seg_upd(pw2, sta, 0)
        d[sta] = 0
        cdef int u, v, w
        cdef int e_i
        while self.seg_min[1] < self.inf:
            u = self.seg_pos[1]
            if vis[u]:
                continue
            vis[u] = 1
            self.seg_upd(pw2, u, self.inf)
            for i in range(pos[u], pos[u + 1]):
                e_i = e[i]
                v, w = y[e_i], z[e_i]
                if d[u] + w < d[v]:
                    d[v] = d[u] + w
                    self.seg_upd(pw2, v, d[v])
