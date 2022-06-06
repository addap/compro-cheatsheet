/// ---------- IMPORTS ----------------
use std::{
    cmp::Ordering,
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
    if let Some(mut last) = s.get(0) {
        for x in s {
            if x < last {
                return false;
            }

            last = x;
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

fn dfs_depth(adj: &Vec<Vec<usize>>, start: usize, max_depth: u64) -> Vec<bool> {
    let n = adj.len();
    let mut border = VecDeque::with_capacity(n);
    border.push_back((start, 0));
    let mut visited: Vec<bool> = vec![false; n];
    visited[start] = true;

    while !border.is_empty() {
        let (v, vd) = border.pop_back().unwrap();

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

fn dfs_rec(adj: &Vec<Vec<usize>>, visited: &mut Vec<bool>, start: usize, max_depth: u64) {
    visited[start] = true;

    if max_depth == 0 {
        return;
    }

    for neighbor in &adj[start] {
        if !visited[*neighbor] {
            dfs_rec(adj, visited, *neighbor, max_depth - 1);
        }
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
