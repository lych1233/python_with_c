## 使用C加速Python：Tutorial & Evaluation for Numba JIT, Cython, Ctypes + Cpp on Pyhton with C Modules

一个简单的测试与教程，包含两个简单的任务：质数统计的Eratosthenes筛法，以及最短路dijkstra算法（包含系统内置heap，和手写线段树加速两个版本）

测试环境：2019年个人笔记本，win11, i7-9750H CPU @ 2.60GHz, NVIDIA GeForce GTX 1050；运行结果为连续测试5次取最优

包环境：python=3.8.5, numba=0.56.4, numpy=1.91.1



### Vanilla Python

#### 代码

代码：

```pythonimport time
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
print("time to count primes", time.time() - time_before_prime)
```



#### 结果

运行结果

```bash
number of primes 809281
time to count primes 31.729369640350342

============================================================

distance from 1 to 123456 = 325821946
distance from 2 to 61728 = 228774322
distance from 3 to 41152 = 118858507
distance from 4 to 30864 = 108589930
distance from 5 to 24691 = 297591540
time for dijkstra with heap 24.423399686813354

distance from 1 to 123456 = 325821946
distance from 2 to 61728 = 228774322
distance from 3 to 41152 = 118858507
distance from 4 to 30864 = 108589930
distance from 5 to 24691 = 297591540
time for dijkstra with segment tree 165.15777349472046
```



### Numba

体验：
- 具有非常完整的报错信息，很容易调试以及找到运行时间瓶颈，并且具有比较完善的文档支持
- 只有受numba支持的部分能够加速，无法完全享受到c和c++的一些特性（如C++ STL）



#### 代码

```python
import time
import heapq

import numpy as np
from numba import njit
from numba.typed import List


prime_N = 12345678
graph_V = 123456
graph_E = 1234567


# always use njit for efficieny (numba will check for validity)
# cache=True to avoid slow running for the first time
@njit(cache=True)
def count_primes(n, non_prime):
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
prime_counter = count_primes(prime_N, non_prime)
print("number of primes", prime_counter)
print("time to count primes", time.time() - time_before_prime)


print("\n" + "=" * 60 + "\n")
data = np.load("data.npy")
X, Y, Z = data[0], data[1], data[2]


@njit(cache=True)
def dijkstra_heap(n, m, sta, x, y, z, pos, e, vis, d, Q):
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
    d[sta] = 0
    while Q:
        _, u = heapq.heappop(Q)
        if vis[u]:
            continue
        vis[u] = 1
        for i in range(pos[u], pos[u + 1]):
            e_i = e[i]
            v, w = y[e_i], z[e_i]
            if d[u] + w < d[v]:
                d[v] = d[u] + w
                heapq.heappush(Q, (d[v], np.int64(v)))

time_before_dijkstra = time.time()
for sta in range(1, 1 + 5):
    pos = np.zeros(graph_V + 9, dtype=int)
    e = np.zeros(graph_E + 9, dtype=int)
    vis = np.zeros(graph_V + 9, dtype=bool)
    d = np.ones(graph_V + 9, dtype=np.int64) * int(1e9)
    Q = List([(0, sta)])
    dijkstra_heap(graph_V, graph_E, sta, X, Y, Z, pos, e, vis, d, Q)
    tar = graph_V // sta
    print(f"distance from {sta} to {tar} = {d[tar]}")
print("time for dijkstra with (python built-in) heap", time.time() - time_before_dijkstra)
print()


@njit(cache=True)
def seg_init(seg_min, seg_pos, n, pw2):
    for i in range(1, 1 + n):
        seg_pos[i | pw2] = i
    for i in range((n | pw2) >> 1, 1, -1):
        if seg_min[i << 1] < seg_min[i << 1 | 1]:
            seg_pos[i] = seg_pos[i << 1]
        else:
            seg_pos[i] = seg_pos[i << 1| 1]

@njit(cache=True)
def seg_upd(seg_min, seg_pos, pw2, x, y):
    k = x | pw2
    seg_min[k] = y
    while k > 1:
        k >>= 1
        l = k << 1 | (seg_min[k << 1 | 1] < seg_min[k << 1])
        seg_min[k], seg_pos[k] = seg_min[l], seg_pos[l]

@njit(cache=True)
def dijkstra_segment(seg_min, seg_pos, pw2, n, m, sta, x, y, z, pos, e, vis, d):
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

    seg_upd(seg_min, seg_pos, pw2, sta, 0)
    d[sta] = 0
    while seg_min[1] < int(1e9):
        u = seg_pos[1]
        if vis[u]:
            continue
        vis[u] = 1
        seg_upd(seg_min, seg_pos, pw2, u, int(1e9))
        for i in range(pos[u], pos[u + 1]):
            e_i = e[i]
            v, w = y[e_i], z[e_i]
            if d[u] + w < d[v]:
                d[v] = d[u] + w
                seg_upd(seg_min, seg_pos, pw2, v, d[v])

time_before_segment = time.time()
for sta in range(1, 1 + 5):
    seg_min = np.ones(graph_V * 4 + 9, dtype=int) * int(1e9)
    seg_pos = np.zeros(graph_V * 4 + 9, dtype=int)
    pw2 = 1
    while pw2 <= graph_V:
        pw2 <<= 1
    seg_init(seg_min, seg_pos, graph_V, pw2)
    pos = np.zeros(graph_V + 9, dtype=int)
    e = np.zeros(graph_E + 9, dtype=int)
    vis = np.zeros(graph_V + 9, dtype=bool)
    d = np.ones(graph_V + 9, dtype=int) * int(1e9)
    dijkstra_segment(seg_min, seg_pos, pw2, graph_V, graph_E, sta, X, Y, Z, pos, e, vis, d)
    tar = graph_V // sta
    print(f"distance from {sta} to {tar} = {d[tar]}")
print("time for dijkstra with segment tree", time.time() - time_before_segment)
print()
```


#### 结果

```bash
number of primes 809281
time to count primes 1.362342357635498

============================================================

distance from 1 to 123456 = 325821946
distance from 2 to 61728 = 228774322
distance from 3 to 41152 = 118858507
distance from 4 to 30864 = 108589930
distance from 5 to 24691 = 297591540
time for dijkstra with (python built-in) heap 1.5404198169708252

distance from 1 to 123456 = 325821946
distance from 2 to 61728 = 228774322
distance from 3 to 41152 = 118858507
distance from 4 to 30864 = 108589930
distance from 5 to 24691 = 297591540
time for dijkstra with segment tree 0.5512020587921143
```



### Cython

体验：
- 功能非常强大，兼容性非常强
- 需要手动预编译（也可以使用[import pyximport](https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html)来避免），因此更适合加速模块不需要频繁修改的场景
- 编译后生成的C语言代码并不容易阅读
- 因为只要语法正确就能成功运行，因此难以发现运行时的性能瓶颈；而numba则有严格的检查机制，但与之相对，numba就无法像cython一样灵活；所以需要使用一些工具来检查cython的运行效率



#### 代码

首先写一个cython_utils.pyx来存放需要加速的模块（如果想使用c++特性，注意在开头指定需要c++编译器）：

```cython
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
```

然后需要一个cython_setup.py文件来进行编译：

```python
from setuptools import setup
from Cython.Build import cythonize

import numpy as np

setup(
    ext_modules=cythonize("cython_utils.pyx",
        language_level=3,
    ),
    include_dirs=[np.get_include()]
)
```

运行：

```
python cython_setup.py build_ext --inplace
```

得到可执行的文件，之后直接在python中import使用即可

```python
import time

import numpy as np
import cython_utils as c_utils


prime_N = 12345678
graph_V = 123456
graph_E = 1234567


time_before_prime = time.time()
non_prime = np.zeros(prime_N + 1, dtype=int)
prime_counter = c_utils.count_primes(prime_N, non_prime)
print("numer of primes", prime_counter)
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
```



#### 结果

```bash
number of primes 809281
time to count primes 1.0351805686950684

============================================================

distance from 1 to 123456 = 325821946
distance from 2 to 61728 = 228774322
distance from 3 to 41152 = 118858507
distance from 4 to 30864 = 108589930
distance from 5 to 24691 = 297591540
time for dijkstra with (C++ STL) heap 0.46610355377197266

distance from 1 to 123456 = 325821946
distance from 2 to 61728 = 228774322
distance from 3 to 41152 = 118858507
distance from 4 to 30864 = 108589930
distance from 5 to 24691 = 297591540
time for dijkstra with segment tree 0.5703625679016113
```



#### 如何检查cython的运行瓶颈

方法一：使用cythonize -a来可视化哪些部分对python的依赖更重，即：

```bash
cythonize -a [name of cython modules].pyx
```

方法二：使用line-profiler；line-profiler原本是用于检测python单行运行结果，但也可以用于cython，在要测试的cython文件开头加上

```cython
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
```

重新编译.pyx文件后在python中调用（以质数筛为例），请注意line-profiler本身会影响运行速度

```python
import line_profiler

profile = line_profiler.LineProfiler(func.count_primes)
prime_counter = profile.runcall(func.count_primes, prime_N, non_prime)
profile.print_stats()
```

结果：

```
Total time: 47.7958 s
File: cython_functions.pyx
Function: count_primes at line 11

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    11                                           def count_primes(int n, int [:] non_prime):
    12                                               cdef int i, j
    13         1         13.0     13.0      0.0      for i in range(2, n):
    14  12345676   27642822.0      2.2      5.8          for j in range(2, n // i):
    15 172632734  419334093.0      2.4     87.7              non_prime[i * j] = 1
    16         1          2.0      2.0      0.0      cdef int prime_counter = 0
    17         1          2.0      2.0      0.0      for i in range(2, n):
    18  12345676   29113887.0      2.4      6.1          if not non_prime[i]:
    19    809281    1866973.0      2.3      0.4              prime_counter += 1
    20         1        216.0    216.0      0.0      return prime_counter
```

### Ctypes

体验：基本等同于写c++，如果只需要传常见的类型（int, np.ndarray），则只需简单修改接口即可

C++部分的代码：

```cpp
#include <cstdio>
#include <queue>
#define pii std::pair<int, int>
#define mpr std::make_pair

// make the interface compatible with C
extern "C" {

int count_primes(int n, int *non_prime) {
    int i, j;
    for (i = 2; i < n; i++)
        for (j = 2; j < n / i; j++)
            non_prime[i * j] = 1;
    int prime_counter = 0;
    for (i = 2; i < n; i++)
        if (!non_prime[i])
            prime_counter += 1;
    return prime_counter;
}

void dijkstra_heap(int n, int m, int sta, int *x, int *y, int *z, int *pos, int *e, int *vis, int *d) {
    int i;
    for (i = 1; i <= m; i++)
        pos[x[i]] += 1;
    for (i = 1; i <= n; i++)
        pos[i] += pos[i - 1];
    for (i = 1; i <= m; i++) {
        e[pos[x[i]]] = i;
        pos[x[i]] -= 1;
    }
    for (i = 1; i <= n; i++)
        pos[i] += 1;
    pos[n + 1] = m + 1;

    std::priority_queue< pii > Q;
    Q.push(mpr(0, sta));
    d[sta] = 0;
    int u, v, w, e_i;
    pii o;
    while (!Q.empty()) {
        o = Q.top();
        u = o.second;
        Q.pop();
        if (vis[u]) continue;
        vis[u] = 1;
        for (i = pos[u]; i < pos[u + 1]; i++) {
            e_i = e[i];
            v = y[e_i];
            w = z[e_i];
            if (d[u] + w < d[v]) {
                d[v] = d[u] + w;
                Q.push(mpr(-d[v], v));
            }
        }
    }
}

int seg_min[40000000], seg_pos[40000000];
const int inf = (int)(1e9);

void seg_init(int n, int pw2) {
    int i;
    for (i = 0; i < 4 * n; i++) {
        seg_min[i] = inf;
        seg_pos[i] = 0;
    }
    for (i = 1; i <= n; i++)
        seg_pos[i | pw2] = i;
    for (i = (n | pw2) >> 1; i; i--)
        if (seg_min[i << 1] < seg_min[i << 1 | 1])
            seg_pos[i] = seg_pos[i << 1];
        else
            seg_pos[i] = seg_pos[i << 1| 1];
}

void seg_upd(int pw2, int x, int y) {
    int k = x | pw2;
    seg_min[k] = y;
    int l;
    while (k > 1) {
        k >>= 1;
        l = k << 1 | (seg_min[k << 1 | 1] < seg_min[k << 1]);
        seg_min[k] = seg_min[l];
        seg_pos[k] = seg_pos[l];
    }
}

void dijkstra_segment(int pw2, int n, int m, const int sta, int *x, int *y, int * z, int *pos, int *e, int *vis, int *d) {
    int i;
    for (i = 1; i <= m; i++)
        pos[x[i]] += 1;
    for (i = 1; i <= n; i++)
        pos[i] += pos[i - 1];
    for (i = 1; i <= m; i++) {
        e[pos[x[i]]] = i;
        pos[x[i]] -= 1;
    }
    for (i = 1; i <= n; i++)
        pos[i] += 1;
    pos[n + 1] = m + 1;

    seg_upd(pw2, sta, 0);
    d[sta] = 0;
    int u, v, w;
    int e_i;
    while (seg_min[1] < inf) {
        u = seg_pos[1];
        if (vis[u]) continue;
        vis[u] = 1;
        seg_upd(pw2, u, inf);
        for (i = pos[u]; i < pos[u + 1]; i++) {
            e_i = e[i];
            v = y[e_i];
            w = z[e_i];
            if (d[u] + w < d[v]) {
                d[v] = d[u] + w;
                seg_upd(pw2, v, d[v]);
            }
        }
    }
}

}
```

然后使用

```bash
g++ -fPIC -shared -o ctypes_utils.so ctypes_utils.cpp
```

将其编译为.so文件后，使用ctypes来调用

```python
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
```

运行结果：

```bash
number of primes 809281
time to count primes 1.3410861492156982

============================================================

distance from 1 to 123456 = 325821946
distance from 2 to 61728 = 228774322
distance from 3 to 41152 = 118858507
distance from 4 to 30864 = 108589930
distance from 5 to 24691 = 297591540
time for dijkstra with (C++ STL) heap 1.0098302364349365

distance from 1 to 123456 = 325821946
distance from 2 to 61728 = 228774322
distance from 3 to 41152 = 118858507
distance from 4 to 30864 = 108589930
distance from 5 to 24691 = 297591540
time for dijkstra with segment tree 0.7336339950561523
```


### 结论

- numba（nopython模式）和cython（无瓶颈）的速度一致
- numba的语法和python基本完全相同，几乎可以直接添加@jit来完成加速
- cython的写法更接近c，但依然与python差别不大，语法学习成本不高
- cython需要在命令行中额外编译
- cython几乎总是能加速，但需要小心在微小的地方成为性能瓶颈，这一部分学习成本较高

如何选择：
- 如果对c和c++熟悉，优先选择功能强大的cython（cython语法本身其实也不难学）
- 如果只会python语法，或者需要频繁改写需要加速的模块，可以使用更方便的numba
- 不推荐使用ctypes从头搭建project，建议只用于和现成的c模块交互