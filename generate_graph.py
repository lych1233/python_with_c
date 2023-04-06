import numpy as np


np.random.seed(0)
graph_V = 123456
graph_E = 1234567
max_edge_length = int(1e6)

proposed_d = np.random.choice(graph_V * max_edge_length // 100, graph_V)
proposed_d.sort()
proposed_d[0] = 0

shortcut = 0
x = np.random.choice(graph_V, graph_E, replace=True)
y = np.random.choice(graph_V, graph_E, replace=True)
y = x + np.abs(y - x) % 100 * np.sign(y - x)

def sample_edge():
    a, b = 0, 0
    while a == b:
        a, b = np.random.randint(graph_V), np.random.randint(graph_V)
    return a, b

for i in range(graph_V - 1):
    x[i] = i
    y[i] = i + 1

for i in range(graph_V - 1, graph_E):
    if shortcut < 3 and np.random.rand() < 1e-3:
        shortcut += 1
        x[i], y[i] = sample_edge()
    if x[i] == y[i]:
        if x[i] + 1 < graph_V:
            y[i] = x[i] + 1
        else:
            y[i] = x[i] - 1

z = max_edge_length - np.clip(max_edge_length - np.abs(proposed_d[x] - proposed_d[y]), 0, max_edge_length) * (0.8 + 0.2 * np.random.rand(graph_E))
z = z.astype(int) + 1
for i in range(graph_V - 1):
    z[i] = max_edge_length
p = np.random.permutation(graph_V - 1) + 1
p = np.concatenate((np.zeros(1, dtype=int), p))
x, y = p[x] + 1, p[y] + 1

data = np.stack([x, y, z]) # [3, E]
data = np.concatenate((np.zeros((3, 1), dtype=int), data), axis=1)
np.save("data", data)