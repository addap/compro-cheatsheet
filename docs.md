## Tests
cargo run --bin XXX < ./testfile.in

## SSSP
single source shortest path

|                | BFS     | Dijkstra          | Bellman-Ford          | Flo✓d Warshall    |
|-|-|-|-|-|
| Running Time   | O(V+E) | O((V+E)\*log V) | O(VE) | O(V^3) |
| Max Size       | V, E <= 10^7 | V, E <= 10^6 | V * E <= 10^7 | V <= 500 |
| Unweighted     | ✓ | ✓ | ✓ | ✓ |
| Weighted       | ✗ | ✓ | ✓ | ✓ |
| Neg Weights    | ✗ | ✗ | ✓ | ✓ |
| Sparse (E ~ V) | O(V) | O(V*log V) | O(V^2) | O(V^3) |
| Dense (E ~ V^2)| O(V^2) | O(V^2 * log V) | O(V^3) | O(V^3) |
