// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic two-dimensional mesh over genuine measured coordinates.

use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug)]
pub(super) struct Vertex {
    pub tokens: u32,
    pub kv: u32,
    pub latency_ms: f64,
}

type Point = [f64; 2];
type Triangle = [usize; 3];

#[derive(Debug)]
pub(super) struct Mesh {
    vertices: Vec<Vertex>,
    triangles: Vec<Triangle>,
}

impl Mesh {
    pub fn new(mut vertices: Vec<Vertex>) -> Option<Self> {
        vertices.sort_by_key(|vertex| (vertex.tokens, vertex.kv));
        if vertices.len() < 3 {
            return None;
        }
        let origin = [
            vertices.first()?.tokens as f64,
            vertices.iter().map(|v| v.kv).min()? as f64,
        ];
        let scale = [
            vertices.last()?.tokens as f64 - origin[0],
            vertices.iter().map(|v| v.kv).max()? as f64 - origin[1],
        ];
        if scale.contains(&0.0) {
            return None;
        }
        let points: Vec<Point> = vertices
            .iter()
            .map(|v| {
                [
                    (v.tokens as f64 - origin[0]) / scale[0],
                    (v.kv as f64 - origin[1]) / scale[1],
                ]
            })
            .collect();
        let hull = convex_hull(&vertices);
        let mut triangles: Vec<Triangle> = hull[1..]
            .windows(2)
            .map(|edge| [hull[0], edge[0], edge[1]])
            .collect();
        // Start with the exact measured hull. Inserting a site splits its one
        // containing triangle, or both triangles when it lies on a shared edge.
        // This avoids finite synthetic enclosing vertices cutting off thin hulls.
        for site in 0..vertices.len() {
            if hull.contains(&site) {
                continue;
            }
            let mut boundary = BTreeMap::<(usize, usize), usize>::new();
            triangles.retain(|triangle| {
                if !(0..3)
                    .all(|i| exact_turn(&vertices, triangle[i], triangle[(i + 1) % 3], site) >= 0)
                {
                    return true;
                }
                for (a, b) in [
                    (triangle[0], triangle[1]),
                    (triangle[1], triangle[2]),
                    (triangle[2], triangle[0]),
                ] {
                    *boundary.entry((a.min(b), a.max(b))).or_default() += 1;
                }
                false
            });
            for ((a, b), count) in boundary {
                if count != 1 {
                    continue;
                }
                let orientation = exact_turn(&vertices, a, b, site);
                if orientation > 0 {
                    triangles.push([a, b, site]);
                } else if orientation < 0 {
                    triangles.push([b, a, site]);
                }
            }
        }
        improve_edges(&vertices, &points, &mut triangles)?;
        Some(Self {
            vertices,
            triangles,
        })
    }

    pub fn support(&self, tokens: f64, kv: f64) -> Option<Vec<(Vertex, f64)>> {
        let point = [tokens, kv];
        if point.iter().any(|value| !value.is_finite()) {
            return None;
        }
        for triangle in &self.triangles {
            let [a, b, c] = triangle.map(|index| self.vertices[index]);
            let area = exact_turn(&self.vertices, triangle[0], triangle[1], triangle[2]) as f64;
            let weights = [
                query_turn(b, c, point) / area,
                query_turn(c, a, point) / area,
                query_turn(a, b, point) / area,
            ];
            // Strict inclusion prevents boundary holds and extrapolation.
            if weights.iter().all(|weight| (0.0..=1.0).contains(weight)) {
                return Some(
                    triangle
                        .iter()
                        .zip(weights)
                        .filter(|(_, weight)| *weight > 0.0)
                        .map(|(&index, weight)| (self.vertices[index], weight))
                        .collect(),
                );
            }
        }
        None
    }
}

fn exact_turn(vertices: &[Vertex], a: usize, b: usize, c: usize) -> i128 {
    let [a, b, c] = [vertices[a], vertices[b], vertices[c]];
    (b.tokens as i128 - a.tokens as i128) * (c.kv as i128 - a.kv as i128)
        - (b.kv as i128 - a.kv as i128) * (c.tokens as i128 - a.tokens as i128)
}

fn query_turn(a: Vertex, b: Vertex, point: Point) -> f64 {
    if point
        .iter()
        .all(|v| (0.0..=u32::MAX as f64).contains(v) && v.fract() == 0.0)
    {
        return ((b.tokens as i128 - a.tokens as i128) * (point[1] as i128 - a.kv as i128)
            - (b.kv as i128 - a.kv as i128) * (point[0] as i128 - a.tokens as i128))
            as f64;
    }
    orient(
        [a.tokens as f64, a.kv as f64],
        [b.tokens as f64, b.kv as f64],
        point,
    )
}

fn convex_hull(vertices: &[Vertex]) -> Vec<usize> {
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    for (chain, indices) in [
        (&mut lower, (0..vertices.len()).collect::<Vec<_>>()),
        (&mut upper, (0..vertices.len()).rev().collect()),
    ] {
        for index in indices {
            while chain.len() >= 2
                && exact_turn(
                    vertices,
                    chain[chain.len() - 2],
                    chain[chain.len() - 1],
                    index,
                ) <= 0
            {
                chain.pop();
            }
            chain.push(index);
        }
        chain.pop();
    }
    lower.extend(upper);
    lower
}

fn improve_edges(vertices: &[Vertex], points: &[Point], triangles: &mut [Triangle]) -> Option<()> {
    // A failed numerical convergence stays unsupported. Every intermediate
    // mesh still covers the measured hull and uses only measured vertices.
    for _ in 0..vertices
        .len()
        .saturating_mul(vertices.len())
        .saturating_mul(16)
    {
        let mut edges = BTreeMap::<(usize, usize), Vec<(usize, usize)>>::new();
        for (index, triangle) in triangles.iter().enumerate() {
            for i in 0..3 {
                let (a, b) = (triangle[i], triangle[(i + 1) % 3]);
                edges
                    .entry((a.min(b), a.max(b)))
                    .or_default()
                    .push((index, triangle[(i + 2) % 3]));
            }
        }
        let mut changed = false;
        for ((a, b), sides) in edges {
            let [(first, c), (second, d)] = sides.as_slice() else {
                continue;
            };
            let first_side = exact_turn(vertices, *c, *d, a);
            let second_side = exact_turn(vertices, *c, *d, b);
            if first_side == 0 || second_side == 0 || first_side.signum() == second_side.signum() {
                continue;
            }
            if in_circumcircle(points[*d], triangles[*first].map(|i| points[i])) {
                triangles[*first] = if first_side > 0 {
                    [*c, *d, a]
                } else {
                    [*d, *c, a]
                };
                triangles[*second] = if second_side > 0 {
                    [*c, *d, b]
                } else {
                    [*d, *c, b]
                };
                changed = true;
                break;
            }
        }
        if !changed {
            return Some(());
        }
    }
    None
}

fn orient(a: Point, b: Point, c: Point) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// All stored triangles are counterclockwise. A positive determinant means
/// the inserted site lies strictly inside their circumcircle.
fn in_circumcircle(point: Point, [a, b, c]: [Point; 3]) -> bool {
    let [ax, ay] = [a[0] - point[0], a[1] - point[1]];
    let [bx, by] = [b[0] - point[0], b[1] - point[1]];
    let [cx, cy] = [c[0] - point[0], c[1] - point[1]];
    let terms = [
        (ax * ax + ay * ay) * (bx * cy - by * cx),
        -(bx * bx + by * by) * (ax * cy - ay * cx),
        (cx * cx + cy * cy) * (ax * by - ay * bx),
    ];
    terms.iter().sum::<f64>() > 32.0 * f64::EPSILON * terms.iter().map(|v| v.abs()).sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vertex(tokens: u32, kv: u32, latency_ms: f64) -> Vertex {
        Vertex {
            tokens,
            kv,
            latency_ms,
        }
    }

    fn latency(mesh: &Mesh, tokens: f64, kv: f64) -> Option<f64> {
        let support = mesh.support(tokens, kv)?;
        assert!((support.iter().map(|(_, weight)| weight).sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(
            (support
                .iter()
                .map(|(v, w)| v.tokens as f64 * w)
                .sum::<f64>()
                - tokens)
                .abs()
                < 1e-6
        );
        assert!((support.iter().map(|(v, w)| v.kv as f64 * w).sum::<f64>() - kv).abs() < 1e-6);
        Some(
            support
                .iter()
                .map(|(v, weight)| v.latency_ms * weight)
                .sum(),
        )
    }

    #[test]
    fn triangle_weights_and_sloped_domain_are_bounded() {
        let mesh = Mesh::new(vec![
            vertex(0, 0, 10.0),
            vertex(100, 0, 20.0),
            vertex(0, 100, 30.0),
        ])
        .unwrap();
        // Quarter of each edge vertex and half the origin gives 17.5ms.
        assert_eq!(latency(&mesh, 25.0, 25.0), Some(17.5));
        assert_eq!(latency(&mesh, 50.0, 50.0), Some(25.0));
        for point in [[51.0, 50.0], [-1.0, 0.0], [0.0, 101.0], [f64::NAN, 0.0]] {
            assert!(mesh.support(point[0], point[1]).is_none());
        }
    }

    #[test]
    fn thin_triangle_retains_its_complete_measured_hull() {
        let mesh = Mesh::new(vec![
            vertex(64, 32640, 10.0),
            vertex(8128, 24576, 50.0),
            vertex(4096, 28544, 30.0),
        ])
        .unwrap();
        // Equal weights are an interior point even for this thin 32K triangle.
        let result = latency(&mesh, 4096.0, 85760.0 / 3.0).unwrap();
        assert!((result - 30.0).abs() < 1e-10);
    }

    #[test]
    fn grid_is_complete_and_affine_with_cocircular_sites() {
        let vertices: Vec<_> = (0..=2)
            .flat_map(|x| {
                (0..=2)
                    .map(move |y| vertex(x * 100, y * 100, 7.0 + 2.0 * x as f64 + 3.0 * y as f64))
            })
            .collect();
        let forward = Mesh::new(vertices.clone()).unwrap();
        let reverse = Mesh::new(vertices.into_iter().rev().collect()).unwrap();
        for x in 0..=32 {
            for y in 0..=32 {
                let (tokens, kv) = (x as f64 * 6.25, y as f64 * 6.25);
                let expected = 7.0 + 2.0 * x as f64 / 16.0 + 3.0 * y as f64 / 16.0;
                let actual = latency(&forward, tokens, kv).unwrap();
                assert!((actual - expected).abs() < 1e-12);
                assert_eq!(latency(&reverse, tokens, kv), Some(actual));
            }
        }
    }

    #[test]
    fn irregular_interior_sites_cover_the_complete_hull() {
        // The independently specified plane is 7 + 0.02*tokens + 0.03*KV.
        let coordinates = [
            (0, 0),
            (200, 0),
            (0, 200),
            (200, 200),
            (1, 1),
            (19, 173),
            (27, 49),
            (51, 195),
            (69, 81),
            (87, 21),
            (99, 103),
            (115, 139),
            (127, 61),
            (141, 189),
            (159, 7),
            (173, 121),
            (199, 199),
        ];
        let vertices = coordinates
            .into_iter()
            .map(|(x, y)| vertex(x, y, 7.0 + 0.02 * x as f64 + 0.03 * y as f64))
            .collect();
        let mesh = Mesh::new(vertices).unwrap();
        for x in 0..=40 {
            for y in 0..=40 {
                let actual = latency(&mesh, (x * 5) as f64, (y * 5) as f64).unwrap();
                assert!((actual - (7.0 + 0.1 * x as f64 + 0.15 * y as f64)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn degenerate_and_large_axes_are_safe() {
        assert!(Mesh::new(vec![vertex(1, 1, 1.0), vertex(2, 2, 2.0)]).is_none());
        let line = Mesh::new(vec![
            vertex(1, 1, 1.0),
            vertex(2, 2, 2.0),
            vertex(3, 3, 3.0),
        ])
        .unwrap();
        assert!(line.support(2.0, 2.0).is_none());
        let large = Mesh::new(vec![
            vertex(0, 0, 1.0),
            vertex(4_000_000_000, 0, 3.0),
            vertex(0, 4_000_000_000, 5.0),
            vertex(4_000_000_000, 4_000_000_000, 7.0),
        ])
        .unwrap();
        assert_eq!(latency(&large, 2_000_000_000.0, 2_000_000_000.0), Some(4.0));
    }
}
