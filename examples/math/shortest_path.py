"""Dijkstra's algorithm on a weighted graph: final distances in the nodes and
the shortest-path tree highlighted."""

import heapq

import pytikz as tz

edges = [
    ("s", "a", 4), ("s", "b", 1), ("b", "a", 2), ("a", "c", 1), ("b", "d", 5),
    ("c", "d", 3), ("c", "e", 6), ("d", "e", 2), ("d", "t", 7), ("e", "t", 3), ("a", "e", 8),
]
positions = {"s": (0, 0), "a": (2.2, 1.4), "b": (2.2, -1.4), "c": (4.6, 1.4), "d": (4.6, -1.4),
             "e": (7.0, 0.2), "t": (9.2, -0.6)}

# Dijkstra from s
adj = {}
for u, v, w in edges:
    adj.setdefault(u, []).append((v, w))
    adj.setdefault(v, []).append((u, w))
dist, parent = {"s": 0}, {}
heap = [(0, "s")]
while heap:
    d, u = heapq.heappop(heap)
    if d > dist[u]:
        continue
    for v, w in adj[u]:
        if d + w < dist.get(v, float("inf")):
            dist[v], parent[v] = d + w, u
            heapq.heappush(heap, (d + w, v))

pic = tz.Picture()
g = tz.Graph(
    {v: f"${v}$" for v in positions},
    [(u, v, w) for u, v, w in edges],
    layout=positions,
    node_size=0.75,
    name="g",
)
for v, u in parent.items():
    g.highlight_edge(u, v, color="Orange", width=2.2)
path, v = [], "t"
while v != "s":
    path.append(v)
    v = parent[v]
path.append("s")
for v in path:
    g.highlight_node(v, "Orange")
pic.add(g)

for v, d in dist.items():
    above = positions[v][1] > 1
    pic.node(f"$d={d}$", at=g.node(v).north if above else g.node(v).south,
             anchor="south" if above else "north", font=r"\scriptsize", text_color="Ink!70")

pic.node(r"shortest path $s \to t$: " + r" $\to$ ".join(f"${v}$" for v in reversed(path)) +
         f" (length {dist['t']})", at=(4.6, -2.6), font=r"\small")

if __name__ == "__main__":
    pic.render("shortest_path.pdf")
