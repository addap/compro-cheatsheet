/// ---------- IMPORTS ----------------
use std::{
    cmp::{max, min, Ordering},
    collections::{HashMap, HashSet, VecDeque},
    io::{self, Read},
};

/// ---------- READING INPUT ------------

/// next_token() Makro um whitespace-separated Zeug zu lesen
fn tokenize() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();

    let mut tokens = input.split_ascii_whitespace();
    macro_rules! next_token {
        ( $t:ty ) => {
            tokens.next().unwrap().parse::<$t>().unwrap()
        };
    }
}

/// Zeilenbasiert lesen
fn linize() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();

    let lines = input.split_terminator('\n');
}

/// Dichten Graph mit Adjazenzmatrix lesen
fn dense_graph() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();

    let mut tokens = input.split_ascii_whitespace();
    macro_rules! next_token {
        ( $t:ty ) => {
            tokens.next().unwrap().parse::<$t>().unwrap()
        };
    }

    let n = next_token!(usize);
    let m = next_token!(usize);

    let mut adj = vec![vec![0 as u32; n]; n];

    for _ in 0..m {
        let start = next_token!(usize);
        let end = next_token!(usize);

        adj[start][end] = 1;
        adj[end][start] = 1;
    }
}

/// Lichten Graph lesen mit Adjazenzlisten
fn sparse_graph() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();

    let mut tokens = input.split_ascii_whitespace();
    macro_rules! next_token {
        ( $t:ty ) => {
            tokens.next().unwrap().parse::<$t>().unwrap()
        };
    }

    let n = next_token!(usize);
    let m = next_token!(usize);

    let mut adj = vec![Vec::new(); n];

    for _ in 0..m {
        let start = next_token!(usize);
        let end = next_token!(usize);

        // TODO remove if not 1-based
        adj[start - 1].push(end - 1);
        adj[end - 1].push(start - 1);
    }
}

/// --------- MISC -------------
fn diverge() {
    loop {}
}

fn is_in_order<T: Ord>(s: &Vec<T>) -> bool {
    if let Some(mut prev) = s.get(0) {
        for x in s {
            if x < prev {
                return false;
            }

            prev = x;
        }
    }

    true
}

mod binary_search {
    use super::*;

    type N = u64;
    // NOTE: max exclusive
    fn binary_search<F>(mut min: N, mut max: N, target: N, f: F) -> Option<N>
    where
        F: Fn(N) -> N,
    {
        while min < max {
            let middle = min + ((max - min) / 2);

            match f(middle).cmp(&target) {
                Ordering::Less => {
                    min = middle + 1;
                }
                Ordering::Equal => return Some(middle),
                Ordering::Greater => {
                    max = middle;
                }
            }
        }

        None
    }

    // NOTE: max exclusive
    fn binary_search_max<F>(mut min: N, mut max: N, f: F) -> Option<N>
    where
        F: Fn(N) -> bool,
    {
        let mut success = None;
        while min < max {
            let middle = min + ((max - min) / 2);

            if f(middle) {
                success = Some(middle);
                min = middle + 1;
            } else {
                max = middle;
            }
        }

        success
    }
}

mod permutation {
    fn do_for_perm(perm: &mut Vec<u64>, visited: &mut [bool], f: &mut impl FnMut(&Vec<u64>) -> ()) {
        let n = visited.len();

        if perm.len() >= n {
            f(perm);
        } else {
            for i in 0..n {
                if visited[i] {
                    continue;
                } else {
                    perm.push(i as u64);
                    visited[i] = true;
                    do_for_perm(perm, visited, f);
                    perm.pop();
                    visited[i] = false;
                }
            }
        }
    }

    fn do_for_perms_of(n: usize, f: &mut impl FnMut(&Vec<u64>) -> ()) {
        let mut perm = Vec::with_capacity(n);
        let mut visited = vec![false; n];

        do_for_perm(&mut perm, &mut visited, f);
    }
}

/// return a mapping [sum |-> count] of all subset sums of the given vector.
/// for meet in the middle
fn subset_sums(v: Vec<i64>) -> HashMap<i64, i64> {
    assert!(v.len() <= 64);

    let mut sums = HashMap::new();
    for bits in 0u64..(1 << v.len()) {
        let mut sum = 0;
        for bit in 0..v.len() {
            if bits & (1 << bit) != 0 {
                sum += v[bit];
            }
        }
        let count_sum = sums.entry(sum).or_insert(0);
        *count_sum += 1;
    }
    sums
}

/// return a set of all subset sums
fn subset_sums_set(v: Vec<i64>) -> HashSet<i64> {
    assert!(v.len() <= 64);

    let mut sums = HashSet::new();
    for bits in 0u64..(1 << v.len()) {
        let mut sum = 0;
        for bit in 0..v.len() {
            if bits & (1 << bit) != 0 {
                sum += v[bit];
            }
        }
        sums.insert(sum);
    }
    sums
}

// longest increasing subsequence (n <= 5000)
fn lis() {
    // for i in 0..n {
    //     let dpi0 = next_token!(u64);

    //     let mut mx = 0;
    //     for j in 0..i {
    //         if dp[j].0 < dpi0 && dp[j].1 > mx {
    //             mx = dp[j].1
    //         }
    //     }

    //     dp.push((dpi0, mx + 1));
    // }
}

fn lis_binary() {
    // let mut A: Vec<i64> = Vec::with_capacity(n);
    // let mut g = Vec::new();
    // g(l) := smallest a s.t. exists increasing subseq with length l
    // g.push(i64::MIN);

    // let mut dp = Vec::with_capacity(n);

    // for i in 0..n {
    //     let Ai = next_token!(i64);
    //     A.push(Ai);

    //     // find longest lis that can be extended by Ai
    //     let fi = 1 + binary_search_max(0, g.len(), |l| g[l] < Ai).unwrap();

    //     dp.push(fi);

    //     if fi >= g.len() {
    //         // new longest lis found, need to extend g
    //         assert!(fi == g.len());
    //         g.push(Ai);
    //     } else {
    //         // set to the smallest end-value of a lis of length fi
    //         g[fi] = min(g[fi], Ai);
    //     }
    // }
}

/// Knapsack using dp
fn knapsack() {
    // let N = next_token!(usize);
    //     let W = next_token!(i64);

    //     let mut profits = Vec::with_capacity(N);
    //     let mut weights = Vec::with_capacity(N);
    //     for _ in 0..N {
    //         profits.push(next_token!(i64));
    //         weights.push(next_token!(i64));
    //     }

    //     let mut dp = vec![vec![0; N]; (W + 1) as usize];
    //     for v in 0..=W {
    //         for i in 0..N {
    //             dp[v as usize][i] = if i == 0 {
    //                 if v < weights[0] {
    //                     0
    //                 } else {
    //                     profits[0]
    //                 }
    //             } else {
    //                 if v < weights[i] {
    //                     dp[v as usize][i - 1]
    //                 } else {
    //                     max(
    //                         dp[v as usize][i - 1],
    //                         profits[i] + dp[(v - weights[i]) as usize][i - 1],
    //                     )
    //                 }
    //             }
    //         }
    //     }

    //     println!("{}", dp[W as usize][N - 1]);
}

/// -------------- PARSER --------------------
mod parser {
    use std::iter::Peekable;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum Token {
        And,
        Or,
        LBrace,
        RBrace,
        Comma,
        Nat(u64),
        X,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum NT {
        Tree,
        Nat,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]

    enum Expr {
        Mul(Box<Expr>, Box<Expr>),
        Add(Box<Expr>, Box<Expr>),
        Leaf(u64),
        Var,
    }

    fn tokenize(s: &str) -> Vec<Token> {
        let mut tokens = Vec::with_capacity(s.len());

        let mut chars = s.chars().peekable();
        while let Some(c) = chars.next() {
            let token = match c {
                '*' => Token::And,
                '+' => Token::Or,
                '(' => Token::LBrace,
                ')' => Token::RBrace,
                ',' => Token::Comma,
                'X' => Token::X,
                '0'..='9' => {
                    let mut n: u64 = c.to_digit(10).unwrap() as u64;

                    while let Some(c2) = chars.peek() {
                        match c2 {
                            '0'..='9' => {
                                n = 10 * n + c2.to_digit(10).unwrap() as u64;
                                chars.next().unwrap();
                            }
                            _ => break,
                        }
                    }
                    Token::Nat(n)
                }
                _ => panic!("Unknown token {}", c),
            };
            tokens.push(token)
        }

        tokens
    }

    fn expect(token: Token, tokens: &mut impl Iterator<Item = Token>) {
        assert!(token == tokens.next().unwrap());
    }

    fn parse_rec<I: Iterator<Item = Token>>(nt: NT, tokens: &mut Peekable<I>) -> Expr {
        let token = tokens.peek().unwrap();

        match (nt, token) {
            (NT::Tree, Token::And) | (NT::Tree, Token::Or) => {
                let op = tokens.next().unwrap();

                expect(Token::LBrace, tokens);
                let subtree1 = parse_rec(NT::Tree, tokens);
                expect(Token::Comma, tokens);
                let subtree2 = parse_rec(NT::Tree, tokens);
                expect(Token::RBrace, tokens);

                if op == Token::And {
                    Expr::Mul(Box::new(subtree1), Box::new(subtree2))
                } else {
                    Expr::Add(Box::new(subtree1), Box::new(subtree2))
                }
            }
            (NT::Tree, Token::Nat(_)) => parse_rec(NT::Nat, tokens),
            (NT::Tree, Token::X) => {
                tokens.next().unwrap();
                Expr::Var
            }
            (NT::Nat, Token::Nat(n)) => {
                let res = Expr::Leaf(*n);
                tokens.next().unwrap();
                res
            }
            _ => unreachable!(),
        }
    }

    fn eval(expr: &Expr, x: u64) -> u64 {
        match expr {
            Expr::Mul(e1, e2) => eval(&*e1, x) * eval(&*e2, x),
            Expr::Add(e1, e2) => eval(&*e1, x) + eval(&*e2, x),
            Expr::Leaf(n) => *n,
            Expr::Var => x,
        }
    }
}

/// -------------- GRAPHS --------------------
fn bfs(adj: &Vec<Vec<usize>>, start: usize) -> Vec<bool> {
    let n = adj.len();
    let mut border = VecDeque::with_capacity(n);
    border.push_back(start);
    let mut visited: Vec<bool> = vec![false; n];
    visited[start] = true;

    while !border.is_empty() {
        let v = border.pop_front().unwrap();

        for neighbor in &adj[v] {
            if !visited[*neighbor] {
                border.push_back(*neighbor);
                visited[*neighbor] = true;
            }
        }
    }

    visited
}

fn bfs_depth(adj: &Vec<Vec<usize>>, start: usize, max_depth: u64) -> Vec<bool> {
    let n = adj.len();
    let mut border = VecDeque::with_capacity(n);
    border.push_back((start, 0));
    let mut visited: Vec<bool> = vec![false; n];
    visited[start] = true;

    while !border.is_empty() {
        let (v, vd) = border.pop_front().unwrap();

        if vd < max_depth {
            for neighbor in &adj[v] {
                if !visited[*neighbor] {
                    border.push_back((*neighbor, vd + 1));
                    visited[*neighbor] = true;
                }
            }
        }
    }

    visited
}

fn dfs(adj: &Vec<Vec<usize>>, start: usize) -> Vec<bool> {
    let n = adj.len();
    let mut border = VecDeque::with_capacity(n);
    border.push_back(start);
    let mut visited: Vec<bool> = vec![false; n];
    visited[start] = true;

    while !border.is_empty() {
        let v = border.pop_back().unwrap();

        for neighbor in &adj[v] {
            if !visited[*neighbor] {
                border.push_back(*neighbor);
                visited[*neighbor] = true;
            }
        }
    }

    visited
}

fn dfs_rec(adj: &Vec<Vec<usize>>, visited: &mut Vec<bool>, start: usize) {
    visited[start] = true;

    for neighbor in &adj[start] {
        if !visited[*neighbor] {
            dfs_rec(adj, visited, *neighbor);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VType {
    Unvisited,
    Exploring,
    Finished,
}

// returns true if cycle exists
fn dfs_cycle(adj: &Vec<Vec<usize>>, visited: &mut Vec<VType>, start: usize) -> bool {
    visited[start] = VType::Exploring;

    for &neighbor in &adj[start] {
        if visited[neighbor] == VType::Unvisited {
            if dfs_cycle(adj, visited, neighbor) {
                return true;
            }
        } else if visited[neighbor] == VType::Exploring {
            return true;
        }
    }

    return false;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Color {
    Red,
    Blue,
}

impl Color {
    fn next(self) -> Self {
        match self {
            Color::Red => Color::Blue,
            Color::Blue => Color::Red,
        }
    }
}

fn dfs_color(adj: &Vec<Vec<usize>>, colors: &mut Vec<Option<Color>>, start: usize) -> bool {
    for &neighbor in &adj[start] {
        if colors[neighbor].is_none() {
            colors[neighbor] = Some(colors[start].unwrap().next());
        } else if colors[start].unwrap() == colors[neighbor].unwrap() {
            return false;
        }
    }

    true
}

// bipartite check
fn color_graph(adj: &Vec<Vec<usize>>, start: usize) -> bool {
    let n = adj.len();
    let mut colors = vec![None; n];
    colors[start] = Some(Color::Red);

    dfs_color(adj, &mut colors, start)
}

mod dijkstra {
    use std::{cmp::Ordering, collections::BinaryHeap};

    #[derive(Debug, PartialEq, Eq)]
    struct DEntry {
        v: usize,
        cost: u64,
    }

    impl Ord for DEntry {
        fn cmp(&self, other: &Self) -> Ordering {
            other
                .cost
                .cmp(&self.cost)
                .then_with(|| self.v.cmp(&other.v))
        }
    }

    impl PartialOrd for DEntry {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct Edge {
        v: usize,
        cost: u64,
    }

    fn dijkstra(adj: &Vec<Vec<Edge>>, start: usize, goal: usize) -> Option<u64> {
        let n = adj.len();
        let mut dist = vec![u64::MAX; n];
        let mut pq = BinaryHeap::with_capacity(n);

        dist[start] = 0;
        pq.push(DEntry { v: start, cost: 0 });

        while let Some(DEntry { v, cost }) = pq.pop() {
            if v == goal {
                return Some(cost);
            }

            if cost > dist[v] {
                continue;
            }

            for edge in &adj[v] {
                let next = DEntry {
                    v: edge.v,
                    cost: edge.cost + cost,
                };

                if next.cost < dist[next.v] {
                    dist[next.v] = next.cost;
                    pq.push(next);
                }
            }
        }
        None
    }
}

mod dijkstra_rich {
    use std::{cmp::Ordering, collections::BinaryHeap};

    #[derive(Debug, PartialEq, Eq)]
    struct DEntry {
        v: usize,
        cost: u64,
        lds: usize,
    }

    impl Ord for DEntry {
        fn cmp(&self, other: &Self) -> Ordering {
            other
                .cost
                .cmp(&self.cost)
                .then_with(|| self.v.cmp(&other.v))
        }
    }

    impl PartialOrd for DEntry {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    #[derive(Debug, Clone, Copy)]
    enum ET {
        LongDistance,
        Regional,
    }

    #[derive(Debug, Clone, Copy)]
    struct Edge {
        v: usize,
        cost: u64,
        ty: ET,
    }

    fn dijkstra(adj: &Vec<Vec<Edge>>, start: usize, goal: usize, t: usize) -> Option<u64> {
        let n = adj.len();
        // dist keeps track of the length depending on the number of long distance trains
        let mut dist = vec![vec![u64::MAX; t + 1]; n];
        let mut pq = BinaryHeap::with_capacity(n);

        dist[start][0] = 0;
        // we start with no long distance trains used
        pq.push(DEntry {
            v: start,
            cost: 0,
            lds: 0,
        });

        while let Some(e) = pq.pop() {
            // we must never put something with lds > t in pq
            if e.v == goal {
                return Some(e.cost);
            }

            if e.cost > dist[e.v][e.lds] {
                continue;
            }

            for edge in &adj[e.v] {
                let next = DEntry {
                    v: edge.v,
                    cost: edge.cost + e.cost,
                    lds: match edge.ty {
                        ET::LongDistance => e.lds + 1,
                        ET::Regional => e.lds,
                    },
                };

                if next.lds > t {
                    continue;
                }

                if next.cost < dist[next.v][next.lds] {
                    dist[next.v][next.lds] = next.cost;
                    pq.push(next);
                }
            }
        }
        None
    }
}

mod floyd_warshall {
    use super::*;

    // NOTE: adj is adjancency matrix
    fn floyd_warshall(adj: &mut Vec<Vec<usize>>) {
        let n = adj.len();

        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    adj[i][j] = min(adj[i][j], adj[i][k] + adj[k][j]);
                }
            }
        }
    }
}

mod bellman_ford {
    use super::*;

    // NOTE: adj is weighted adjancency list
    fn bellman_ford(adj: &Vec<Vec<(usize, i64)>>, start: usize) {
        let n = adj.len();
        let mut dist = vec![i64::MAX; n];
        dist[start] = 0;

        // relax everything n-1 times
        for i in 0..n - 1 {
            for v in 0..n {
                for &(neighbor, w) in &adj[v] {
                    dist[neighbor] = min(dist[neighbor], dist[v] + w);
                }
            }
        }
    }
}

mod toposort {

    fn topo(adj: &Vec<Vec<usize>>, ts: &mut Vec<usize>) -> Option<()> {
        let n = adj.len();
        let mut visited = vec![false; n];

        for i in 0..n {
            if !visited[i] {
                let mut stack = vec![false; n];
                stack[i] = true;
                if let None = dfs_rec(adj, &mut visited, ts, &mut stack, i) {
                    return None;
                }
            }
        }
        ts.reverse();
        Some(())
    }

    fn dfs_rec(
        adj: &Vec<Vec<usize>>,   // adjacency list
        visited: &mut Vec<bool>, // which nodes have been visited during the toposort
        ts: &mut Vec<usize>,     // the final topological sort. Add each node when we leave it.
        stack: &mut Vec<bool>,   // which nodes have been visited during this dfs run
        start: usize,
    ) -> Option<()> {
        visited[start] = true;

        for neighbor in &adj[start] {
            if !visited[*neighbor] {
                stack[*neighbor] = true;
                if dfs_rec(adj, visited, ts, stack, *neighbor).is_none() {
                    return None;
                }
                stack[*neighbor] = false;
            } else if stack[*neighbor] {
                // got a cycle
                return None;
            }
        }

        ts.push(start);
        Some(())
    }
}

mod bridges {
    // Articulation point: A vertex whose removal (as well as its incident edges) increases the
    // number of connected components is called an Articulation Point.
    // Bridge: An edge whose removal increases the number of connected
    // components is called a Bridge.

    use super::*;

    const UNVISITED: i64 = -1;

    fn ap_and_bridges(adj: &Vec<Vec<usize>>) -> (Vec<usize>, Vec<(usize, usize)>) {
        let n = adj.len();
        let mut dfs_counter = 0;

        let mut aps = Vec::new();
        let mut bridges = Vec::new();

        let mut dfs_num = vec![UNVISITED; n];
        let mut dfs_min = vec![UNVISITED; n];
        let mut dfs_parent = vec![-1; n];

        for i in 0..n {
            if dfs_num[i] == UNVISITED {
                let dfs_root = i;
                let mut root_children = 0;

                dfs(
                    i,
                    adj,
                    dfs_root,
                    &mut root_children,
                    &mut dfs_counter,
                    &mut dfs_num,
                    &mut dfs_min,
                    &mut dfs_parent,
                    &mut aps,
                    &mut bridges,
                );

                if root_children > 1 {
                    aps.push(i);
                }
            }
        }

        (aps, bridges)
    }

    fn dfs(
        start: usize,
        adj: &Vec<Vec<usize>>,
        dfs_root: usize,
        root_children: &mut usize,
        dfs_counter: &mut i64,
        dfs_num: &mut Vec<i64>,
        dfs_min: &mut Vec<i64>,
        dfs_parent: &mut Vec<i64>,
        aps: &mut Vec<usize>,
        bridges: &mut Vec<(usize, usize)>,
    ) {
        dfs_min[start] = *dfs_counter;
        dfs_num[start] = *dfs_counter;
        *dfs_counter += 1;

        for &neighbor in &adj[start] {
            if dfs_num[neighbor] == UNVISITED {
                // tree edge
                dfs_parent[neighbor] = start as i64;
                if start == dfs_root {
                    *root_children += 1;
                }

                dfs(
                    neighbor,
                    adj,
                    dfs_root,
                    root_children,
                    dfs_counter,
                    dfs_num,
                    dfs_min,
                    dfs_parent,
                    aps,
                    bridges,
                );

                if dfs_num[start] <= dfs_min[neighbor] && start != dfs_root {
                    aps.push(start);
                }
                if dfs_num[start] < dfs_min[neighbor] {
                    bridges.push((start, neighbor));
                }

                dfs_min[start] = min(dfs_min[start], dfs_min[neighbor]);
            } else if neighbor as i64 != dfs_parent[start] {
                // back edge
                dfs_min[start] = min(dfs_min[start], dfs_num[neighbor]);
            }
        }
    }
}

mod tarjan {
    use super::*;

    const UNVISITED: i64 = -1;

    // find strongly connected components
    fn sccs(adj: &Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        let n = adj.len();
        let mut dfs_counter = 0;

        let mut sccs = Vec::new();
        let mut stack = Vec::with_capacity(n);

        let mut dfs_num = vec![UNVISITED; n];
        let mut dfs_min = vec![UNVISITED; n];
        let mut dfs_parent = vec![-1; n];

        for i in 0..n {
            if dfs_num[i] == UNVISITED {
                dfs(
                    i,
                    adj,
                    &mut dfs_counter,
                    &mut dfs_num,
                    &mut dfs_min,
                    &mut dfs_parent,
                    &mut sccs,
                    &mut stack,
                );
            }
            assert!(stack.is_empty());
        }

        // reverse so that return value is immediately a topological ordering
        sccs.reverse();
        sccs
    }

    fn sccs_to_dag(n: usize, adj: &Vec<Vec<usize>>, comps: &Vec<Vec<usize>>) {
        let mut repr_to_comp = HashMap::with_capacity(comps.len());
        let mut comp_to_repr = HashMap::with_capacity(n);

        for comp in comps {
            let repr = *comp.get(0).unwrap();

            repr_to_comp.insert(repr, comp);
            for v in comp {
                comp_to_repr.insert(*v, repr);
            }
        }
        let mut comp_adj = vec![Vec::new(); n];
        for v in 0..n {
            let repr_v = *comp_to_repr.get(&v).unwrap();

            for neighbor in &adj[v] {
                let repr_neighbor = *comp_to_repr.get(&neighbor).unwrap();

                if repr_neighbor != repr_v {
                    comp_adj[repr_v].push(repr_neighbor);
                }
            }
        }
    }

    fn dfs(
        start: usize,
        adj: &Vec<Vec<usize>>,
        dfs_counter: &mut i64,
        dfs_num: &mut Vec<i64>,
        dfs_min: &mut Vec<i64>,
        dfs_parent: &mut Vec<i64>,
        sccs: &mut Vec<Vec<usize>>,
        stack: &mut Vec<usize>,
    ) {
        dfs_min[start] = *dfs_counter;
        dfs_num[start] = *dfs_counter;
        *dfs_counter += 1;

        stack.push(start);
        let stackpos = stack.len() - 1;

        for &neighbor in &adj[start] {
            if dfs_num[neighbor] == UNVISITED {
                // tree edge

                dfs(
                    neighbor,
                    adj,
                    dfs_counter,
                    dfs_num,
                    dfs_min,
                    dfs_parent,
                    sccs,
                    stack,
                );

                dfs_min[start] = min(dfs_min[start], dfs_min[neighbor]);
            } else if stack.contains(&neighbor) {
                // back edge
                dfs_min[start] = min(dfs_min[start], dfs_num[neighbor]);
            }
        }

        // when backtracking, check if dfs_num == dfs_min, in which case we have a scc.
        if dfs_min[start] == dfs_num[start] {
            let scc = stack.drain(stackpos..stack.len()).collect();
            sccs.push(scc);
        }
    }
}

mod dag {
    use super::*;

    /// shortest path from sources, but can be used for apsp by subtracting
    /// also possible: longest path, counting paths
    fn sssp(adj: &Vec<Vec<(usize, u64)>>, ts: &Vec<usize>) -> Vec<u64> {
        let n = adj.len();
        let mut dist = vec![u64::MAX; n];

        for &v in ts {
            for &(neighbor, w) in &adj[v] {
                dist[neighbor] = min(dist[neighbor], dist[v] + w);
            }
        }

        dist
    }
}

/// Eulerian trail, 0 or 2 vertices have odd degree (the start and end). All others have even degree.
/// We also check if all edges belong to one component, i.e. all vertices not in that component have 0 edges between them.
mod euler {
    use std::{
        cmp::{max, min, Ordering},
        collections::{HashMap, HashSet, VecDeque},
        io::{self, Read},
    };

    fn main() {
        let mut input = String::new();
        io::stdin().read_to_string(&mut input).unwrap();

        let mut tokens = input.split_ascii_whitespace();
        macro_rules! next_token {
            ( $t:ty ) => {
                tokens.next().unwrap().parse::<$t>().unwrap()
            };
        }

        let n = next_token!(usize);
        let m = next_token!(usize);

        let mut adj = vec![Vec::new(); n];
        let mut indeg = vec![0usize; n];

        for _ in 0..m {
            let start = next_token!(usize);
            let end = next_token!(usize);

            adj[start - 1].push(end - 1);
            indeg[end - 1] += 1;
        }

        // check in/out-degree of all vertices
        let mut euler_possible = true;
        for i in 0..n {
            if i == 0 {
                if adj[i].len() != indeg[i] + 1 {
                    euler_possible = false;
                    break;
                }
            } else if i == n - 1 {
                if adj[i].len() != indeg[i] - 1 {
                    euler_possible = false;
                    break;
                }
            } else if adj[i].len() != indeg[i] {
                euler_possible = false;
                break;
            }
        }

        if euler_possible {
            // try to walk the cycle, if at end and not all nodes are visited, we have a separate component;
            let mut visited = vec![false; n];
            dfs_rec(&adj, &mut visited, 0);

            // vertices with nonzero degree are allowed belong to other component
            let all_visited = visited
                .iter()
                .enumerate()
                .all(|(i, &is_visited)| is_visited || (adj[i].len() == 0 && indeg[i] == 0));
            if !all_visited {
                println!("impossible");
            } else {
                println!("possible");
            }
        } else {
            println!("impossible");
        }
    }

    fn dfs_rec(adj: &Vec<Vec<usize>>, visited: &mut Vec<bool>, start: usize) {
        visited[start] = true;

        for neighbor in &adj[start] {
            if !visited[*neighbor] {
                dfs_rec(adj, visited, *neighbor);
            }
        }
    }

    // NOTE: cycle is reversed
    // for connected graph, cycle.len() == adj.len() + 1
    fn euler_cycle(adj: &mut Vec<Vec<usize>>, cycle: &mut Vec<usize>, start: usize) {
        if let Some(neighbor) = adj[start].pop() {
            euler_cycle(adj, cycle, neighbor);
            assert!(adj[start].is_empty());
        }

        cycle.push(start);
    }
}

// --------- QUERYING ---------
mod querying {
    use std::io::{self, Read};

    const WILD: bool = true;
    const TAME: bool = false;

    // put everything into buckets of size sqrt(N)
    fn precompute(colonies: &[bool]) -> (Vec<u64>, usize) {
        let bucket_len = (colonies.len() as f64).sqrt().ceil() as usize;

        let mut buckets = Vec::new();
        for (i, b) in colonies.iter().enumerate() {
            if i % bucket_len == 0 {
                buckets.push(0);
            }
            if *b == WILD {
                let idx = buckets.len() - 1;
                buckets[idx] += 1;
            }
        }

        (buckets, bucket_len)
    }

    // count number of WILDs between i & j inclusive
    fn count_wilds(
        buckets: &[u64],
        bucket_len: usize,
        colonies: &[bool],
        i: usize,
        j: usize,
    ) -> u64 {
        let mut count = 0;

        let mut x = i;
        loop {
            if x > j {
                break;
            }

            // j is inclusive so if it's the last element of a bucket we still count the whole bucket
            if x % bucket_len == 0 && j - x >= bucket_len - 1 {
                count += buckets[x / bucket_len];
                x += bucket_len;
            } else {
                if colonies[x] == WILD {
                    count += 1;
                }
                x += 1;
            }
        }

        count
    }

    fn update_behavior(
        buckets: &mut [u64],
        bucket_len: usize,
        colonies: &mut [bool],
        i: usize,
        b: bool,
    ) {
        let bi = i / bucket_len;

        if colonies[i] == WILD && b == TAME {
            buckets[bi] -= 1;
            colonies[i] = TAME;
        } else if colonies[i] == TAME && b == WILD {
            buckets[bi] += 1;
            colonies[i] = WILD;
        }
    }

    fn main() {
        let mut input = String::new();
        io::stdin().read_to_string(&mut input).unwrap();

        let mut tokens = input.split_ascii_whitespace();
        macro_rules! next_token {
            ( $t:ty ) => {
                tokens.next().unwrap().parse::<$t>().unwrap()
            };
        }

        let n = next_token!(usize);
        let q = next_token!(usize);

        let mut colonies = Vec::with_capacity(n);
        for _ in 0..n {
            let behavior = match next_token!(u64) % 10 {
                1 | 3 | 5 | 7 | 9 => WILD,
                _ => TAME,
            };
            colonies.push(behavior);
        }

        let (mut buckets, bucket_len) = precompute(&colonies);

        for _ in 0..q {
            match tokens.next().unwrap() {
                "Q" => {
                    let i = next_token!(usize) - 1;
                    let j = next_token!(usize) - 1;

                    println!("{}", count_wilds(&buckets, bucket_len, &colonies, i, j));
                }
                "U" => {
                    let i = next_token!(usize) - 1;
                    let b = match next_token!(u64) % 10 {
                        1 | 3 | 5 | 7 | 9 => WILD,
                        _ => TAME,
                    };

                    update_behavior(&mut buckets, bucket_len, &mut colonies, i, b);
                }
                _ => unreachable!(),
            }
        }
    }
}

mod lca_binarylift {
    //////////////////
    /// Usage: Get the tree as adjacency list (directed edges towards children only), n (num nodes), root node
    /// Call precompute
    /// Call lca
    /// Profit
    /// //////////////

    //      node        adj-list                  height/depth        immediate parent
    fn dfs(u: usize, graph: &Vec<Vec<usize>>, h: &mut Vec<usize>, p: &mut Vec<usize>) {
        for &v in &graph[u] {
            if v != p[u] {
                p[v] = u;
                h[v] = h[u] + 1;
                dfs(v, graph, h, p);
            }
        }
    }

    pub struct Lca {
        pub graph: Vec<Vec<usize>>,
        // height such that the root has height 0
        pub h: Vec<usize>,
        // p[i][u] == 2^i-th ancestor of u; p[0][u] is the immediate parent of u
        pub p: Vec<Vec<usize>>,
    }

    impl Lca {
        // kth ancestor of node "u", 0th ancestor is u itself, 1st is direct parent etc
        pub fn kth_ancestor(&self, u: usize, mut k: usize) -> usize {
            let mut res = u;
            // do binary lifting: decompose k into a binary string; for each 1 in it, we know the parent
            for i in (0..(self.p.len())).rev() {
                if k >= (1 << i) {
                    // k has the "i"th one
                    k -= (1 << i);
                    res = self.p[i][res];
                }
            }
            res
        }

        // O(N* log(N))
        // n == number of nodes in graph
        // graph == adj. list
        pub fn precompute(n: usize, root: usize, graph: Vec<Vec<usize>>) -> Lca {
            let l = ((graph.len() as f64).log2().ceil() as usize) + 1;
            // depths
            let mut h: Vec<usize> = vec![0; n];
            let mut p: Vec<Vec<usize>> = vec![vec![0; n]; l];
            p[0][root] = root;
            // fill depths and immediate parents
            dfs(root, &graph, &mut h, &mut p[0]);

            for i in 1..l {
                for j in 0..n {
                    // for example, the 8th parent == 4th parent of the 4th parent
                    p[i][j] = p[i - 1][p[i - 1][j]];
                }
            }

            Lca { graph, h, p }
        }

        // O(log(N))
        pub fn lca(&self, mut u: usize, mut v: usize) -> usize {
            if self.h[v] > self.h[u] {
                std::mem::swap(&mut u, &mut v);
            }

            // lift u to the level of v
            u = self.kth_ancestor(u, self.h[u] - self.h[v]);
            if u == v {
                // v is some ancestor of u, so their lca is v
                return v;
            }

            // do binary lifting, jump as much as possible such that u and v are still in separate branches
            for i in (0..(self.p.len())).rev() {
                if self.p[i][u] != self.p[i][v] {
                    u = self.p[i][u];
                    v = self.p[i][v];
                }
            }

            self.p[0][u]
        }
    }
}

mod unionfind {

    pub struct UnionFind {
        pub parent: Vec<usize>,
        rank: Vec<usize>,
    }

    impl UnionFind {
        pub fn new(n: usize) -> UnionFind {
            UnionFind {
                parent: (0..n).collect(),
                rank: vec![0; n],
            }
        }
        pub fn find(&mut self, x: usize) -> usize {
            if self.parent[x] == x {
                return x;
            } else {
                self.parent[x] = self.find(self.parent[x]);
                return self.parent[x];
            }
        }

        pub fn union(&mut self, x1: usize, x2: usize) {
            let i = self.find(x1);
            let j = self.find(x2);
            if i != j {
                if self.rank[i] > self.rank[j] {
                    self.parent[j] = i;
                } else {
                    self.parent[i] = j;
                    if self.rank[i] == self.rank[j] {
                        self.rank[j] += 1;
                    }
                }
            }
        }
    }
}

// calculates the MST (minimum spanning tree) in O(N*Log(N)). Graph given as edge list.
// Returns the edges that are part of MST (the resulting set is symmetric since undirected edges are considered)
// After the algorithm, the input edges are sorted
fn kruskal_minimum_spanning_tree(
    num_nodes: usize,
    weighted_edges: &mut Vec<(usize, usize, i64)>,
) -> HashSet<(usize, usize)> {
    let mut uf = unionfind::UnionFind::new(num_nodes);
    let mut mst = HashSet::new();
    weighted_edges.sort_by_key(|(_, _, v)| *v);
    for &(s, d, _) in weighted_edges.iter() {
        if uf.find(s) != uf.find(d) {
            uf.union(s, d);

            mst.insert((s, d));
            mst.insert((d, s));
        }
    }

    mst
}

mod kmp {
    // String searching: KMP
    // if m = |t| and n = |s|, we have:
    // precalculation in O(m)
    // finding matches in O(n)
    // O(n+m) (where n and m are the string sizes)

    // calculates the longest suffix prefix
    // but can also calculate "shorter" suffix prefixes if the given one is not "ok"
    fn advanced_lsp(t: &str, is_ok: impl Fn(usize) -> bool) -> usize {
        let mut lsp = vec![0; t.len()];
        let mut i = 1;
        let mut prev = 0;
        let m = t.len();
        while i < m {
            if t.as_bytes()[i] == t.as_bytes()[prev] {
                prev += 1;
                lsp[i] = prev;
                i += 1;
            } else if prev == 0 {
                lsp[i] = 0;
                i += 1;
            } else {
                prev = lsp[prev - 1];
            }
        }

        let mut res = lsp[t.len() - 1];
        while !is_ok(res) {
            res = lsp[res - 1];
        }
        return res;
    }

    // longest suffix prefix array: if t = t_0...t_(m-1), then
    // lsp[i] is the longest (strict) suffix prefix of t_0....t_i
    // that is: lsp[m-1] is the LSP of the full string t
    fn lsp(t: &str) -> Vec<usize> {
        let mut lsp = vec![0; t.len()];
        let mut i = 1;
        let mut prev = 0;
        let m = t.len();
        while i < m {
            if t.as_bytes()[i] == t.as_bytes()[prev] {
                prev += 1;
                lsp[i] = prev;
                i += 1;
            } else if prev == 0 {
                lsp[i] = 0;
                i += 1;
            } else {
                prev = lsp[prev - 1];
            }
        }

        return lsp;
    }

    fn kmp(s: &str, t: &str) -> Option<usize> {
        let lsp = lsp(&t);
        let n = s.len();
        let m = t.len();
        let mut start: i64 = 0;
        let mut l: i64 = 0;
        while start + l < (n as i64) {
            while l >= (m as i64) || s.as_bytes()[(start + l) as usize] != t.as_bytes()[l as usize]
            {
                if l == 0 {
                    start += 1;
                    l -= 1;
                    break;
                }
                let skip = l - (lsp[(l - 1) as usize] as i64);
                start += skip;
                l -= skip;
            }
            l += 1;
            if l as usize == m {
                return Some(start as usize);
            }
        }
        return None;
    }
}

mod mathstuff {

    // change usize to your type of desire accordingly
    fn gcd(mut a: usize, mut b: usize) -> usize {
        if a < b {
            swap(&mut a, &mut b);
        }
        if b == 0 {
            return a;
        }
        let r = a % b;
        gcd(b, r)
    }

    // returns (x,y, gcd(a,b)) such that a*x + b*y = gcd(a,b)
    fn extended_euclidean(mut a: i32, mut b: i32) -> (i32, i32, i32) {
        let mut x1 = 1;
        let mut y1 = 0;
        let mut x2 = 0;
        let mut y2 = 1;
        let (mut d, mut r) = (a / b, a % b);
        while r > 0 {
            (x1, y1, x2, y2, a, b) = (x2, y2, x1 - d * x2, y1 - d * y2, b, r);
            (d, r) = (a / b, a % b);
        }

        (x2, y2, b)
    }

    fn fexp(n: usize, k: usize, p: usize) -> usize {
        if k == 0 {
            return 1;
        } else if k % 2 == 0 {
            let r = fexp(n, k / 2, p);
            return (r * r) % p;
        } else {
            return (n * fexp(n, k - 1, p)) % p;
        }
    }

    fn modinv(n: usize, p: usize) -> usize {
        fexp(n, p - 2, p)
    }

    fn fac_modp(n: usize, p: usize) -> usize {
        let mut res = 1;
        for i in 2..=n {
            res = (res * i) % p;
        }
        res
    }

    fn binom_modp(n: usize, mut k: usize, p: usize) -> usize {
        // we absolutely need to make sure that k <= (n/2)
        if 2 * k > n {
            k = n - k;
        }
        let mut res = 1;
        for i in (n - k + 1)..=n {
            res = (res * i) % p;
        }
        (res * modinv(fac_modp(k, p), p)) % p
    }
}

/*
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
*/
