# ternary-loss

**Loss functions that understand Z₃ arithmetic — because {-1, 0, +1} has structure you shouldn't ignore.**

[![Tests](https://img.shields.io/badge/tests-32%20passing-brightgreen)]()
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## Why This Exists

Standard loss functions assume continuous predictions. Cross-entropy expects logits over real-valued classes. MSE expects real-valued outputs. Hinge loss assumes a continuous margin. When your network operates in ternary space {-1, 0, +1}, these assumptions break.

Worse: Z₃ has *cyclic* arithmetic. In Z₃, 1 + 1 = -1 (it wraps around). The "distance" between -1 and +1 is 1 (same as between 0 and +1). Standard distance metrics don't capture this structure. If you naively compute `(predicted - target)²`, you get wrong answers.

This crate provides loss functions built on proper Z₃ arithmetic — every addition, subtraction, and multiplication uses explicit match arms over the nine possible input pairs.

## The Key Insight

Z₃ is a *field*. Every non-zero element has a multiplicative inverse (-1 × -1 = 1, 1 × 1 = 1). This means you can do real algebra in Z₃ — division, polynomial roots, linear systems — all within the ternary alphabet. The loss functions in this crate exploit that structure: distances are computed via Z₃ subtraction, probabilities respect the three-class topology, and KL divergence operates over the natural {-1, 0, +1} partition.

```rust
pub fn z3_add(a: Ternary, b: Ternary) -> Ternary {
    match (a, b) {
        (-1, -1) => 1,   // wraps! -2 mod 3 = 1
        (-1, 0) => -1,
        (-1, 1) => 0,
        (0, -1) => -1,
        (0, 0) => 0,
        (0, 1) => 1,
        (1, -1) => 0,
        (1, 0) => 1,
        (1, 1) => -1,    // wraps! 2 mod 3 = -1
        _ => panic!("invalid ternary value"),
    }
}
```

No `(a + b + 3) % 3 - 1`. Every case enumerated. Every path auditable. Correct by construction.

## Quick Start

```toml
[dependencies]
ternary-loss = "0.1"
```

```rust
use ternary_loss::*;

// ── Z₃ Arithmetic ──
assert_eq!(z3_add(1, 1), -1);    // wraps in Z₃!
assert_eq!(z3_sub(1, -1), -1);   // 1 - (-1) = 1 + 1 = -1
assert_eq!(z3_mul(-1, -1), 1);
assert_eq!(z3_neg(1), -1);

// ── Ternary Cross-Entropy ──
let logits = vec![
    vec![0.0, 0.0, 10.0],   // strong prediction for class +1
    vec![10.0, 0.0, 0.0],   // strong prediction for class -1
];
let targets = vec![2, 0];   // +1 and -1 (index into {-1, 0, +1})
let loss = ternary_cross_entropy(&logits, &targets);
// Near zero — predictions are correct

// ── Ternary Hinge Loss ──
let scores = vec![vec![-10.0, 0.0, 10.0]]; // class 2 well-separated
let hinge = ternary_hinge_loss(&scores, &[2], 1.0); // 0.0 — margin satisfied

// ── Ternary MSE (Z₃ distance) ──
let predicted = vec![vec![0, 0, 1]];
let target = vec![vec![-1, 0, 1]];
let mse = ternary_mse(&predicted, &target); // 1/3 — one element off by distance 1

// ── Contrastive Loss (embeddings) ──
let emb_a = vec![vec![1, 0, -1]];
let emb_b = vec![vec![1, 0, -1]]; // identical
let cl = ternary_contrastive_loss(&emb_a, &emb_b, &[1.0], 1.0); // 0.0

// ── Triplet Loss ──
let anchors = vec![vec![1, 0, -1]];
let positives = vec![vec![1, 0, -1]]; // same
let negatives = vec![vec![-1, 0, 1]]; // opposite
let tl = ternary_triplet_loss(&anchors, &positives, &negatives, 1.0); // 0.0

// ── KL Divergence ──
let p = vec![vec![0.7, 0.2, 0.1]];
let q = vec![vec![0.1, 0.3, 0.6]];
let kl = ternary_kl_divergence(&p, &q); // > 0 — distributions differ
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Z₃ Arithmetic                      │
│  z3_add, z3_sub, z3_mul, z3_neg, z3_distance        │
└──────────────────────────┬──────────────────────────┘
                           │
    ┌──────────┬───────────┼───────────┬──────────────┐
    │          │           │           │              │
┌───▼───┐ ┌───▼───┐ ┌─────▼────┐ ┌───▼──────┐ ┌────▼─────┐
│Cross- │ │ Hinge │ │   MSE    │ │Contrastive│ │ Triplet  │
│Entropy│ │ Loss  │ │(Z₃ dist)│ │  Loss     │ │  Loss    │
└───────┘ └───────┘ └──────────┘ └───────────┘ └──────────┘
                                                       │
                                                ┌──────▼──────┐
                                                │ KL Divergence│
                                                │ P || Q over  │
                                                │ {-1, 0, +1}  │
                                                └──────────────┘
```

## Loss Function Guide

### Ternary Cross-Entropy

Classification loss over three ternary classes. Applies softmax over the logits for {-1, 0, +1} and computes negative log-likelihood.

**When to use:** Multi-class ternary classification (e.g., sentiment: negative / neutral / positive).

### Ternary Hinge Loss

Margin-based: requires the correct class score to exceed the best incorrect class by at least `margin`.

**When to use:** When you want a *gap* between classes, not just correct classification. Good for training ternary SVM-like models.

### Ternary MSE

Mean squared error using Z₃ distance. Each pair is compared via `z3_sub`; distance is 0 (equal) or 1 (different).

**When to use:** Regression-like tasks in ternary space, or when you need a smooth gradient signal.

### Contrastive Loss

For pairs of ternary embeddings: similar pairs should be close in Z₃ space, dissimilar pairs at least `margin` apart.

**When to use:** Learning ternary embeddings for similarity search, face verification, or retrieval.

### Triplet Loss

Anchor-positive-negative ordering: anchor should be closer to positive than negative by at least `margin`.

**When to use:** Learning to rank ternary embeddings. Face recognition, recommendation systems.

### KL Divergence

KL(P || Q) over the three-element distribution {-1, 0, +1}. Always non-negative (Gibbs' inequality).

**When to use:** Knowledge distillation in ternary space — matching a student ternary network to a teacher distribution.

## Z₃ Algebra Reference

```
Addition (wraps!):
  (-1) + (-1) = 1    (-1) + 0 = -1    (-1) + 1 = 0
  0 + (-1) = -1      0 + 0 = 0        0 + 1 = 1
  1 + (-1) = 0       1 + 0 = 1        1 + 1 = -1

Multiplication (sign rules):
  (-1) × (-1) = 1    0 × anything = 0    1 × 1 = 1

Negation: -(-1) = 1, -0 = 0, -1 = -1
```

Key insight: Z₃ is a **field**. Every non-zero element has a multiplicative inverse. This makes it algebraically richer than Z₄ or Z₆.

## API Reference

### Z₃ Arithmetic

```rust
type Ternary = i32; // -1, 0, or +1

fn z3_add(a: Ternary, b: Ternary) -> Ternary;
fn z3_sub(a: Ternary, b: Ternary) -> Ternary;
fn z3_mul(a: Ternary, b: Ternary) -> Ternary;
fn z3_neg(a: Ternary) -> Ternary;
fn z3_distance(a: Ternary, b: Ternary) -> f64;
fn z3_to_prob(t: Ternary) -> f64;
fn is_ternary(v: i32) -> bool;
```

### Loss Functions

```rust
fn ternary_cross_entropy(logits: &[Vec<f64>], targets: &[usize]) -> f64;
fn ternary_cross_entropy_from_probs(predicted_probs: &[Vec<f64>], targets: &[Ternary]) -> f64;
fn ternary_hinge_loss(scores: &[Vec<f64>], targets: &[usize], margin: f64) -> f64;
fn ternary_mse(predicted: &[Vec<Ternary>], target: &[Vec<Ternary>]) -> f64;
fn ternary_contrastive_loss(a: &[Vec<Ternary>], b: &[Vec<Ternary>], labels: &[f64], margin: f64) -> f64;
fn ternary_triplet_loss(anchors: &[Vec<Ternary>], positives: &[Vec<Ternary>], negatives: &[Vec<Ternary>], margin: f64) -> f64;
fn ternary_kl_divergence(p: &[Vec<f64>], q: &[Vec<f64>]) -> f64;
fn ternary_kl_single(p: &[f64], q: &[f64]) -> f64;
```

## Real-World Example: Ternary Sentiment Analysis

A social media platform classifies posts as negative (-1), neutral (0), or positive (+1). The classifier is a ternary network — all activations in {-1, 0, +1}. During training:

```rust
// Training loop
for (post, label) in training_data {
    let logits = ternary_network_forward(&post); // [score_neg, score_zero, score_pos]
    let target = match label {
        -1 => 0, 0 => 1, 1 => 2, // map to index
    };

    // Combined loss: cross-entropy + hinge for margin
    let ce = ternary_cross_entropy(&[logits.clone()], &[target]);
    let hinge = ternary_hinge_loss(&[logits.clone()], &[target], 1.0);
    let loss = ce + 0.1 * hinge;

    let grad = compute_gradient(loss);
    optimizer.step(&mut params, &grad);
}
```

The hinge loss ensures the network doesn't just get the right answer — it gets it with *margin*, making the ternary decision boundaries robust to noise.

## Performance Characteristics

- **Z₃ arithmetic**: O(1) — single match expression
- **Cross-entropy**: O(n × 3) — softmax + log per sample
- **Hinge loss**: O(n × 3) — find max incorrect per sample
- **MSE**: O(n × d) — pairwise Z₃ subtraction
- **Contrastive/Triplet**: O(n × d) — Z₃ distance computation per pair
- **KL divergence**: O(n × 3) — three-element distribution

Memory: All loss functions are O(1) extra space (they accumulate a scalar).

## Ecosystem Connections

Loss functions are the training signal for the ternary stack:

- [`ternary-optimizer`](https://github.com/SuperInstance/ternary-optimizer) — consumes gradients from these losses
- [`ternary-activation`](https://github.com/SuperInstance/ternary-activation) — outputs that feed into these losses
- [`ternary-matmul`](https://github.com/SuperInstance/ternary-matmul) — the network layers being differentiated
- [`ternary-norm`](https://github.com/SuperInstance/ternary-norm) — normalized activations feeding into losses

## Open Questions

- **Focal loss for ternary**: Standard focal loss down-weights easy examples. A ternary variant could down-weight examples where the correct class logit is already dominant.
- **Z₃-aware regularization**: Weight decay in Z₃ doesn't make sense (you can't make a weight "smaller"). A ternary-aware regularizer could encourage sparsity (more zeros) or balance (equal {-1, 0, +1} distribution).
- **Straight-through estimator integration**: These losses return scalars, but the gradient computation needs the STE from `ternary-activation` to flow through ternary nonlinearities.

## Testing

```bash
cargo test
```

32 tests covering: exhaustive Z₃ arithmetic (all 9 cases for add/sub/mul/neg), cross-entropy for perfect/bad predictions and equal-logits CE = ln(3), uniform distribution CE = ln(3), hinge loss with satisfied/violated margins, MSE for identical/partial/maximum/completely-wrong distance, contrastive loss (similar close, dissimilar far, mixed), triplet loss (perfect/violated/equidistant), KL divergence (identical = 0, non-negative, asymmetric, zero-probability finite), empty-batch inputs (no NaN), single-element inputs, and a README-conformance test locking the documented Quick Start values to the code.

## License

MIT
