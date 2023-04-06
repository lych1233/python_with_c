//gcc -fPIC -shared -o ctypes_utils.so ctypes_utils.c
#include <cstdio>
#include <queue>
#define pii std::pair<int, int>
#define mpr std::make_pair

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
