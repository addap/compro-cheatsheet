use std::{
    cmp::Ordering,
    collections::VecDeque,
    io::{self, Read},
};

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

fn linize() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();

    let lines = input.split_terminator('\n');
}

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

        adj[start - 1].push(end - 1);
        adj[end - 1].push(start - 1);
    }
}

fn diverge() {
    let mut i = 0;
    loop {
        i += 1;
    }
}

type N = u64;
fn binary_search<F>(mut min: N, mut max: N, target: N, f: F) -> Option<u64>
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

// TODO
// set inclusion

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
