# ternary-loss

**Loss functions for ternary neural networks operating in Z₃ = {-1, 0, +1}.**

[![Tests](https://img.shields.io/badge/tests-26%20passing-brightgreen)]()

Standard loss functions assume continuous-valued predictions. When your network operates
in the ternary space Z₃ = {-1, 0, +1}, you need loss functions that respect the algebraic
structure of that space. **ternary-loss** provides mathematically grounded loss functions
where all Z₃ arithmetic uses explicit match arms — no modular arithmetic shortcuts.

## Why Ternary Loss Functions?

In a ternary network, predictions and targets live in {-1, 0, +1}. Using standard loss
functions on ternary values ignores the cyclic structure of Z₃:

```
In Z₃:  1 + 1 = -1   (wraps around!)
        -1 - (-1) = 0
        1 × (-1) = -1
```

This crate provides loss functions built on proper Z₃ arithmetic, ensuring that
distance metrics and probability distributions respect the algebraic structure.

## The Z₃ Rule

**All Z₃ arithmetic uses explicit match arms.** We do NOT use `(a + b + 3) % 3 - 1`
or any modular arithmetic shortcut. Every operation is explicitly enumerated:

```rust
pub fn z3_add(a: Ternary, b: Ternary) -> Ternary {
    match (a, b) {
        (-1, -1) => 1,
        (-1,  0) => -1,
        (-1,  1) => 0,
        ( 0, -1) => -1,
        ( 0,  0) => 0,
        ( 0,  1) => 1,
        ( 1, -1) => 0,
        ( 1,  0) => 1,
        ( 1,  1) => -1,
        _ => panic!("invalid ternary value"),
    }
}
```

This ensures correctness by construction and makes the algebraic structure explicit
and auditable.

## Features

- **Z₃ Arithmetic** — add, subtract, multiply, negate, distance (all via match arms)
- **Ternary Cross-Entropy** — classification loss over {-1, 0, +1} classes
- **Ternary Hinge Loss** — margin-based classification with configurable threshold
- **Ternary MSE** — mean squared error using Z₃ distance
- **Contrastive Loss** — for ternary embeddings: pull similar, push dissimilar
- **Triplet Loss** — anchor-positive-negative ordering in Z₃ space
- **Ternary KL Divergence** — divergence between distributions over ternary classes

## Quick Start

```rust
use ternary_loss::*;

// ── Z₃ Arithmetic ──
assert_eq!(z3_add(1, 1), -1);   // wraps in Z₃
assert_eq!(z3_sub(1, -1), -1);  // 1 - (-1) = 1 + 1 = -1 in Z₃
assert_eq!(z3_mul(-1, -1), 1);
assert_eq!(z3_neg(1), -1);

// ── Ternary Cross-Entropy ──
let logits = vec![
    vec![0.0, 0.0, 10.0],  // strong prediction for class +1
    vec![10.0, 0.0, 0.0],  // strong prediction for class -1
];
let targets = vec![2, 0];  // +1 and -1 (index into {-1, 0, +1})
let ce_loss = ternary_cross_entropy(&logits, &targets);
// Near zero — predictions are correct

// ── Ternary Hinge Loss ──
let scores = vec![
    vec![-10.0, 0.0, 10.0],  // correct class 2 well-separated
];
let hinge = ternary_hinge_loss(&scores, &[2], 1.0);
// Zero — margin is satisfied

// ── Ternary MSE ──
let predicted = vec![vec![0, 0, 1]];
let target = vec![vec![-1, 0, 1]];
let mse = ternary_mse(&predicted, &target);
// 1/3 — one element differs by distance 1

// ── Contrastive Loss ──
let emb_a = vec![vec![1, 0, -1]];
let emb_b = vec![vec![1, 0, -1]]; // identical
let loss = ternary_contrastive_loss(&emb_a, &emb_b, &[1.0], 1.0);
// Zero — similar pair is close

// ── Triplet Loss ──
let anchors = vec![vec![1, 0, -1]];
let positives = vec![vec![1, 0, -1]]; // same as anchor
let negatives = vec![vec![-1, 0, 1]]; // opposite
let triplet = ternary_triplet_loss(&anchors, &positives, &negatives, 1.0);
// Zero — positive is closer than negative

// ── KL Divergence ──
let p = vec![vec![0.7, 0.2, 0.1]];
let q = vec![vec![0.1, 0.3, 0.6]];
let kl = ternary_kl_divergence(&p, &q);
// Positive — distributions differ
```

## Loss Function Details

### Ternary Cross-Entropy

Applies softmax over the three ternary classes {-1, 0, +1} and computes negative
log-likelihood. Works with raw logits or pre-computed probabilities.

**When to use:** Multi-class ternary classification where the output is one of
three ternary values.

### Ternary Hinge Loss

Margin-based loss that requires the score of the correct class to exceed the best
incorrect class by at least `margin`. For ternary classification:

```
loss = max(0, margin - (score_correct - score_best_incorrect))
```

**When to use:** When you want a margin between ternary classes, not just correct
classification.

### Ternary MSE (Z₃ Distance)

Computes mean squared error using Z₃ distance. Each pair of ternary values is
compared using `z3_sub`, and the distance is 0 (equal) or 1 (different):

```
MSE = (1/n) × Σ z3_distance(pred_i, target_i)²
```

**When to use:** Regression-like tasks in ternary space, or when you want a smooth
gradient signal from a ternary target.

### Contrastive Loss

For pairs of ternary embeddings (a, b) with similarity labels:

```
loss_similar = ||z3_dist(a, b)||²
loss_dissimilar = max(0, margin - ||z3_dist(a, b)||)²
```

**When to use:** Learning ternary embeddings where similar items should have similar
ternary representations and dissimilar items should be far apart in Z₃ space.

### Triplet Loss

Given anchor (a), positive (p), and negative (n) ternary embeddings:

```
loss = max(0, d(a, p) - d(a, n) + margin)
```

**When to use:** Learning to rank ternary embeddings, ensuring correct ordering
of distances.

### Ternary KL Divergence

Computes KL(P || Q) between distributions over {-1, 0, +1}:

```
KL(P || Q) = Σ_x∈{-1,0,1} P(x) × ln(P(x) / Q(x))
```

With epsilon smoothing for numerical stability. Always non-negative (Gibbs' inequality).

**When to use:** Matching ternary output distributions, knowledge distillation in
ternary space.

## Z₃ Algebra Reference

| Operation | -1 | 0 | +1 |
|-----------|----|---|-----|
| neg(-1) = 1, neg(0) = 0, neg(1) = -1 |
| add(-1, -1) = 1, add(-1, 0) = -1, add(-1, 1) = 0 |
| add(0, -1) = -1, add(0, 0) = 0, add(0, 1) = 1 |
| add(1, -1) = 0, add(1, 0) = 1, add(1, 1) = -1 |
| mul(-1, -1) = 1, mul(0, anything) = 0, mul(1, 1) = 1 |

Key insight: Z₃ is a field! Every non-zero element has a multiplicative inverse
(-1 × -1 = 1, 1 × 1 = 1), making it algebraically richer than Z₄ or Z₆.

## Research Context

- **Ternary Weight Networks**: Li et al. (2016) — constraining weights to {-1, 0, +1}
- **Trained Ternary Quantization**: Zhu et al. (2017) — learning the ternarization threshold
- **FaceNet**: Schroff et al. (2015) — triplet loss for embeddings
- **Contrastive Learning**: Hadsell et al. (2006) — dimensionality reduction by learning
  an invariant mapping

## Testing

```bash
cargo test
```

26 comprehensive tests covering:
- Exhaustive Z₃ arithmetic (add, sub, mul, neg — all 9 cases each)
- Cross-entropy for perfect/bad predictions
- Cross-entropy from probabilities (uniform → ln(3))
- Hinge loss with satisfied/violated margins
- MSE for identical, partial, and maximum distance
- Contrastive loss: similar close, dissimilar far, mixed
- Triplet loss: perfect ordering, violated ordering, equidistant
- KL divergence: identical (zero), non-negative, asymmetric
- Utility functions: is_ternary, z3_to_prob, z3_distance

## License

MIT
