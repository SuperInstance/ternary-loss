//! # ternary-loss
//!
//! Loss functions for ternary neural networks operating in Z₃ = {-1, 0, +1}.
//!
//! All Z₃ arithmetic uses explicit match arms — no modular arithmetic shortcuts.
//! Each loss function is designed for ternary-valued embeddings, activations, or weights.

// ── Z₃ Arithmetic ────────────────────────────────────────────────────────────

/// Ternary value type: -1, 0, or +1.
pub type Ternary = i32;

/// Validate that a value is ternary (in {-1, 0, +1}).
pub fn is_ternary(v: i32) -> bool {
    matches!(v, -1 | 0 | 1)
}

/// Z₃ addition: explicitly enumerated.
pub fn z3_add(a: Ternary, b: Ternary) -> Ternary {
    match (a, b) {
        (-1, -1) => 1,
        (-1, 0) => -1,
        (-1, 1) => 0,
        (0, -1) => -1,
        (0, 0) => 0,
        (0, 1) => 1,
        (1, -1) => 0,
        (1, 0) => 1,
        (1, 1) => -1,
        _ => panic!("invalid ternary value: a={}, b={}", a, b),
    }
}

/// Z₃ subtraction (a - b in Z₃): explicitly enumerated.
pub fn z3_sub(a: Ternary, b: Ternary) -> Ternary {
    match (a, b) {
        (-1, -1) => 0,
        (-1, 0) => -1,
        (-1, 1) => 1,
        (0, -1) => 1,
        (0, 0) => 0,
        (0, 1) => -1,
        (1, -1) => -1, // 1 - (-1) = 1 + 1 = z3_add(1,1) = -1
        (1, 0) => 1,
        (1, 1) => 0,
        _ => panic!("invalid ternary value"),
    }
}

/// Z₃ multiplication: explicitly enumerated.
pub fn z3_mul(a: Ternary, b: Ternary) -> Ternary {
    match (a, b) {
        (-1, -1) => 1,
        (-1, 0) => 0,
        (-1, 1) => -1,
        (0, _) => 0,
        (1, -1) => -1,
        (1, 0) => 0,
        (1, 1) => 1,
        _ => panic!("invalid ternary value"),
    }
}

/// Z₃ negation (additive inverse).
pub fn z3_neg(a: Ternary) -> Ternary {
    match a {
        -1 => 1,
        0 => 0,
        1 => -1,
        _ => panic!("invalid ternary value"),
    }
}

/// Map a ternary value to a probability-like quantity in [0, 1] for loss computation.
pub fn z3_to_prob(t: Ternary) -> f64 {
    match t {
        -1 => 0.0,
        0 => 0.5,
        1 => 1.0,
        _ => panic!("invalid ternary value"),
    }
}

/// Distance between two ternary values in Z₃ (0 if equal, 1 if adjacent, 1 for max).
pub fn z3_distance(a: Ternary, b: Ternary) -> f64 {
    let diff = z3_sub(a, b);
    match diff {
        0 => 0.0,
        1 | -1 => 1.0,
        _ => unreachable!(),
    }
}

// ── Ternary Cross-Entropy ─────────────────────────────────────────────────────

/// Compute ternary cross-entropy loss.
///
/// The "probability" of each ternary class is derived from the predicted logit
/// via a softmax-like mapping over the three classes {-1, 0, +1}.
///
/// `predicted`: raw scores/logits for the three classes, per sample.
/// `target`: ground-truth ternary class index (0=-1, 1=0, 2=+1).
pub fn ternary_cross_entropy(logits: &[Vec<f64>], targets: &[usize]) -> f64 {
    assert_eq!(logits.len(), targets.len());
    let n = logits.len() as f64;
    let mut total = 0.0;

    for (sample_logits, &target) in logits.iter().zip(targets.iter()) {
        assert_eq!(sample_logits.len(), 3, "logits must have 3 entries per sample");
        // Softmax
        let max_logit = sample_logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = sample_logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exp: f64 = exps.iter().sum();
        let prob = exps[target] / sum_exp;
        total -= prob.ln();
    }

    total / n
}

/// Compute ternary cross-entropy with explicit ternary targets.
///
/// `predicted_probs`: probabilities for {-1, 0, +1} per sample (should sum to 1).
/// `target`: ternary target value per sample.
pub fn ternary_cross_entropy_from_probs(
    predicted_probs: &[Vec<f64>],
    targets: &[Ternary],
) -> f64 {
    assert_eq!(predicted_probs.len(), targets.len());
    let n = predicted_probs.len() as f64;
    let mut total = 0.0;

    for (probs, &target) in predicted_probs.iter().zip(targets.iter()) {
        assert_eq!(probs.len(), 3);
        assert!(is_ternary(target));
        let idx = match target {
            -1 => 0,
            0 => 1,
            1 => 2,
            _ => unreachable!(),
        };
        let p = probs[idx].max(1e-12); // clamp for numerical safety
        total -= p.ln();
    }

    total / n
}

// ── Ternary Hinge Loss ────────────────────────────────────────────────────────

/// Compute ternary hinge loss.
///
/// For ternary classification, we treat the score for the correct class and
/// require it to be at least `margin` above the best incorrect class score.
///
/// `scores`: per-sample scores for {-1, 0, +1}.
/// `targets`: target class indices (0, 1, 2).
/// `margin`: the required gap between correct and best incorrect score.
pub fn ternary_hinge_loss(scores: &[Vec<f64>], targets: &[usize], margin: f64) -> f64 {
    assert_eq!(scores.len(), targets.len());
    let n = scores.len() as f64;
    let mut total = 0.0;

    for (sample_scores, &target) in scores.iter().zip(targets.iter()) {
        assert_eq!(sample_scores.len(), 3);
        let correct_score = sample_scores[target];
        let best_incorrect = sample_scores.iter().enumerate()
            .filter(|(i, _)| *i != target)
            .map(|(_, &s)| s)
            .fold(f64::NEG_INFINITY, f64::max);
        total += (margin - (correct_score - best_incorrect)).max(0.0);
    }

    total / n
}

// ── Ternary MSE (Z₃ distance) ────────────────────────────────────────────────

/// Compute ternary Mean Squared Error using Z₃ distance.
///
/// Each element is compared in Z₃ space, and the squared distance is averaged.
/// Uses explicit Z₃ subtraction for correctness.
pub fn ternary_mse(predicted: &[Vec<Ternary>], target: &[Vec<Ternary>]) -> f64 {
    assert_eq!(predicted.len(), target.len());
    let mut total = 0.0;
    let mut count = 0usize;

    for (p_row, t_row) in predicted.iter().zip(target.iter()) {
        assert_eq!(p_row.len(), t_row.len());
        for (&p, &t) in p_row.iter().zip(t_row.iter()) {
            let dist = z3_distance(p, t);
            total += dist * dist;
            count += 1;
        }
    }

    if count == 0 { 0.0 } else { total / count as f64 }
}

// ── Contrastive Loss for Ternary Embeddings ───────────────────────────────────

/// Compute contrastive loss for pairs of ternary embeddings.
///
/// Similar pairs should be close in Z₃ space, dissimilar pairs should be
/// at least `margin` apart.
///
/// `embeddings_a`: first embedding in each pair.
/// `embeddings_b`: second embedding in each pair.
/// `labels`: 1.0 for similar pairs, 0.0 for dissimilar pairs.
/// `margin`: minimum distance for dissimilar pairs.
pub fn ternary_contrastive_loss(
    embeddings_a: &[Vec<Ternary>],
    embeddings_b: &[Vec<Ternary>],
    labels: &[f64],
    margin: f64,
) -> f64 {
    assert_eq!(embeddings_a.len(), embeddings_b.len());
    assert_eq!(embeddings_a.len(), labels.len());
    let n = embeddings_a.len() as f64;
    let mut total = 0.0;

    for i in 0..embeddings_a.len() {
        let dist = ternary_embedding_distance(&embeddings_a[i], &embeddings_b[i]);
        if labels[i] > 0.5 {
            // Similar pair: minimize distance
            total += dist * dist;
        } else {
            // Dissimilar pair: push distance to at least margin
            total += (margin - dist).max(0.0).powi(2);
        }
    }

    total / n
}

/// Compute L2-like distance between two ternary embeddings using Z₃ arithmetic.
fn ternary_embedding_distance(a: &[Ternary], b: &[Ternary]) -> f64 {
    assert_eq!(a.len(), b.len());
    let dist_sq: f64 = a.iter().zip(b.iter())
        .map(|(&ai, &bi)| {
            let d = z3_distance(ai, bi);
            d * d
        })
        .sum();
    dist_sq.sqrt()
}

// ── Triplet Loss ─────────────────────────────────────────────────────────────

/// Compute triplet loss for ternary embeddings.
///
/// Given anchor, positive (same class), and negative (different class) embeddings,
/// the loss encourages the anchor-positive distance to be at least `margin` less
/// than the anchor-negative distance.
pub fn ternary_triplet_loss(
    anchors: &[Vec<Ternary>],
    positives: &[Vec<Ternary>],
    negatives: &[Vec<Ternary>],
    margin: f64,
) -> f64 {
    assert_eq!(anchors.len(), positives.len());
    assert_eq!(anchors.len(), negatives.len());
    let n = anchors.len() as f64;
    let mut total = 0.0;

    for i in 0..anchors.len() {
        let dist_pos = ternary_embedding_distance(&anchors[i], &positives[i]);
        let dist_neg = ternary_embedding_distance(&anchors[i], &negatives[i]);
        total += (dist_pos - dist_neg + margin).max(0.0);
    }

    total / n
}

// ── Ternary KL Divergence ────────────────────────────────────────────────────

/// Compute KL divergence between two ternary distributions.
///
/// Each distribution is represented as probabilities over {-1, 0, +1}
/// (3 values per distribution).
///
/// KL(P || Q) = Σ P(x) × ln(P(x) / Q(x))
///
/// Returns non-negative value. Uses epsilon smoothing for numerical stability.
pub fn ternary_kl_divergence(p: &[Vec<f64>], q: &[Vec<f64>]) -> f64 {
    assert_eq!(p.len(), q.len());
    let epsilon = 1e-12;
    let n = p.len() as f64;
    let mut total = 0.0;

    for (p_dist, q_dist) in p.iter().zip(q.iter()) {
        assert_eq!(p_dist.len(), 3);
        assert_eq!(q_dist.len(), 3);
        for j in 0..3 {
            let p_val = p_dist[j].max(epsilon);
            let q_val = q_dist[j].max(epsilon);
            total += p_val * (p_val / q_val).ln();
        }
    }

    total / n
}

/// KL divergence for a single pair of ternary distributions.
pub fn ternary_kl_single(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), 3);
    assert_eq!(q.len(), 3);
    let epsilon = 1e-12;
    let mut total = 0.0;

    for j in 0..3 {
        let p_val = p[j].max(epsilon);
        let q_val = q[j].max(epsilon);
        total += p_val * (p_val / q_val).ln();
    }

    total
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Z₃ Arithmetic Tests ──

    #[test]
    fn test_z3_add_comprehensive() {
        // Exhaustive verification
        assert_eq!(z3_add(-1, -1), 1);
        assert_eq!(z3_add(-1, 0), -1);
        assert_eq!(z3_add(-1, 1), 0);
        assert_eq!(z3_add(0, -1), -1);
        assert_eq!(z3_add(0, 0), 0);
        assert_eq!(z3_add(0, 1), 1);
        assert_eq!(z3_add(1, -1), 0);
        assert_eq!(z3_add(1, 0), 1);
        assert_eq!(z3_add(1, 1), -1);
    }

    #[test]
    fn test_z3_sub_comprehensive() {
        // a - b in Z₃
        assert_eq!(z3_sub(-1, -1), 0);  // -1 - (-1) = 0
        assert_eq!(z3_sub(-1, 0), -1);  // -1 - 0 = -1
        assert_eq!(z3_sub(-1, 1), 1);   // -1 - 1 = 1 (wraps in Z₃: -2 mod 3 = 1)
        assert_eq!(z3_sub(0, -1), 1);   // 0 - (-1) = 1
        assert_eq!(z3_sub(0, 0), 0);
        assert_eq!(z3_sub(0, 1), -1);   // 0 - 1 = -1
        assert_eq!(z3_sub(1, -1), -1); // 1 - (-1) = 1+1 = -1 in Z₃
        assert_eq!(z3_sub(1, 0), 1);   // 1 - 0 = 1
        assert_eq!(z3_sub(1, 1), 0);   // 1 - 1 = 0
    }

    #[test]
    fn test_z3_mul_comprehensive() {
        assert_eq!(z3_mul(-1, -1), 1);
        assert_eq!(z3_mul(-1, 0), 0);
        assert_eq!(z3_mul(-1, 1), -1);
        assert_eq!(z3_mul(0, -1), 0);
        assert_eq!(z3_mul(0, 0), 0);
        assert_eq!(z3_mul(0, 1), 0);
        assert_eq!(z3_mul(1, -1), -1);
        assert_eq!(z3_mul(1, 0), 0);
        assert_eq!(z3_mul(1, 1), 1);
    }

    #[test]
    fn test_z3_neg_comprehensive() {
        assert_eq!(z3_neg(-1), 1);
        assert_eq!(z3_neg(0), 0);
        assert_eq!(z3_neg(1), -1);
    }

    // ── Cross-Entropy Tests ──

    #[test]
    fn test_cross_entropy_known_targets() {
        // Perfect prediction: high logit for correct class
        let logits = vec![
            vec![0.0, 0.0, 10.0], // target = +1 (idx 2)
            vec![10.0, 0.0, 0.0], // target = -1 (idx 0)
        ];
        let targets = vec![2, 0];
        let loss = ternary_cross_entropy(&logits, &targets);
        assert!(loss >= 0.0, "loss should be non-negative");
        assert!(loss < 0.01, "perfect prediction should have near-zero loss, got {}", loss);
    }

    #[test]
    fn test_cross_entropy_bad_prediction() {
        // Wrong prediction: low logit for correct class
        let logits = vec![
            vec![10.0, 0.0, 0.0], // target = +1 (idx 2), but predicts -1
        ];
        let targets = vec![2];
        let loss = ternary_cross_entropy(&logits, &targets);
        assert!(loss > 1.0, "bad prediction should have high loss, got {}", loss);
    }

    #[test]
    fn test_cross_entropy_from_probs_uniform() {
        // Uniform distribution → loss = -ln(1/3) ≈ 1.099
        let probs = vec![vec![1.0/3.0; 3]];
        let targets = vec![1]; // target = 0, idx 1
        let loss = ternary_cross_entropy_from_probs(&probs, &targets);
        assert!((loss - (3.0_f64).ln()).abs() < 0.01, "uniform CE ≈ ln(3), got {}", loss);
    }

    // ── Hinge Loss Tests ──

    #[test]
    fn test_hinge_loss_satisfied_margin() {
        let scores = vec![
            vec![-10.0, 0.0, 10.0], // correct class 2 has score 10, margin satisfied
        ];
        let targets = vec![2];
        let loss = ternary_hinge_loss(&scores, &targets, 1.0);
        assert_eq!(loss, 0.0, "margin satisfied → zero loss");
    }

    #[test]
    fn test_hinge_loss_violated_margin() {
        let scores = vec![
            vec![5.0, 0.0, 5.5], // correct=2 score=5.5, best_incorrect=5.0, margin=1.0
            // 5.5 - 5.0 = 0.5 < 1.0 → loss = 1.0 - 0.5 = 0.5
        ];
        let targets = vec![2];
        let loss = ternary_hinge_loss(&scores, &targets, 1.0);
        assert!((loss - 0.5).abs() < 1e-10, "margin violated → loss = 0.5, got {}", loss);
    }

    // ── MSE Tests ──

    #[test]
    fn test_mse_identical() {
        let pred = vec![vec![-1, 0, 1]];
        let target = vec![vec![-1, 0, 1]];
        assert_eq!(ternary_mse(&pred, &target), 0.0);
    }

    #[test]
    fn test_mse_computation() {
        let pred = vec![vec![0, 0, 1]];   // predicted
        let target = vec![vec![-1, 0, 1]]; // target
        // z3_distance(0, -1) = z3_sub(0, -1) = 1, so dist=1
        // z3_distance(0, 0) = 0
        // z3_distance(1, 1) = 0
        // MSE = (1² + 0² + 0²) / 3 = 1/3
        let mse = ternary_mse(&pred, &target);
        assert!((mse - 1.0/3.0).abs() < 1e-10, "MSE should be 1/3, got {}", mse);
    }

    #[test]
    fn test_mse_max_distance() {
        let pred = vec![vec![1, -1, 0]];
        let target = vec![vec![-1, 1, 0]];
        // z3_distance(1, -1) = |z3_sub(1,-1)|... need to verify
        // All max distances: each dist=1
        // MSE = (1+1+0)/3 = 2/3
        let mse = ternary_mse(&pred, &target);
        assert!((mse - 2.0/3.0).abs() < 1e-10, "MSE should be 2/3, got {}", mse);
    }

    // ── Contrastive Loss Tests ──

    #[test]
    fn test_contrastive_loss_similar_close() {
        // Identical embeddings → similar loss should be 0
        let a = vec![vec![-1, 0, 1]];
        let b = vec![vec![-1, 0, 1]];
        let labels = vec![1.0]; // similar
        let loss = ternary_contrastive_loss(&a, &b, &labels, 1.0);
        assert_eq!(loss, 0.0, "identical similar pair → zero loss");
    }

    #[test]
    fn test_contrastive_loss_dissimilar_far() {
        // Very different embeddings → dissimilar loss should be 0 if far enough
        let a = vec![vec![-1, -1, -1]];
        let b = vec![vec![1, 1, 1]];
        let labels = vec![0.0]; // dissimilar
        let loss = ternary_contrastive_loss(&a, &b, &labels, 0.5);
        // dist = sqrt(3*1) ≈ 1.73, margin=0.5, max(0.5-1.73, 0)=0
        assert_eq!(loss, 0.0, "far dissimilar pair → zero loss");
    }

    #[test]
    fn test_contrastive_loss_pulls_similar_pushes_different() {
        let emb_a = vec![
            vec![1, 0, -1],
            vec![1, 0, -1],
        ];
        let emb_b = vec![
            vec![1, 0, -1], // identical to a → similar
            vec![-1, 0, 1], // opposite of a → dissimilar
        ];
        let labels = vec![1.0, 0.0];
        let loss = ternary_contrastive_loss(&emb_a, &emb_b, &labels, 10.0);
        // Similar pair: dist=0 → loss=0
        // Dissimilar pair: dist=sqrt(4)=2, margin=10 → (10-2)²=64
        // Average = 32
        assert!(loss > 0.0, "should have non-zero loss from dissimilar pair");
    }

    // ── Triplet Loss Tests ──

    #[test]
    fn test_triplet_loss_perfect_ordering() {
        // anchor closer to positive than negative → zero loss if margin satisfied
        let anchors = vec![vec![1, 0, -1]];
        let positives = vec![vec![1, 0, -1]]; // identical → dist=0
        let negatives = vec![vec![-1, 0, 1]]; // opposite → dist=2
        let loss = ternary_triplet_loss(&anchors, &positives, &negatives, 1.0);
        assert_eq!(loss, 0.0, "d_pos=0 < d_neg=2-1=1 → zero loss");
    }

    #[test]
    fn test_triplet_loss_ordering_violated() {
        let anchors = vec![vec![1, 0, -1]];
        let positives = vec![vec![-1, 0, 1]]; // far from anchor
        let negatives = vec![vec![1, 0, -1]]; // same as anchor
        let loss = ternary_triplet_loss(&anchors, &positives, &negatives, 1.0);
        assert!(loss > 0.0, "d_pos > d_neg → positive loss");
    }

    #[test]
    fn test_triplet_loss_equidistant() {
        let anchors = vec![vec![0, 0, 0]];
        let positives = vec![vec![1, 0, 0]]; // dist=1
        let negatives = vec![vec![-1, 0, 0]]; // dist=1
        let loss = ternary_triplet_loss(&anchors, &positives, &negatives, 1.0);
        // d_pos - d_neg + margin = 1 - 1 + 1 = 1 > 0 → loss = 1
        assert!((loss - 1.0).abs() < 1e-10, "should be 1.0, got {}", loss);
    }

    // ── KL Divergence Tests ──

    #[test]
    fn test_kl_divergence_identical() {
        let p = vec![vec![0.2, 0.3, 0.5]];
        let q = vec![vec![0.2, 0.3, 0.5]];
        let kl = ternary_kl_divergence(&p, &q);
        assert!(kl.abs() < 1e-10, "KL(P||P) should be 0, got {}", kl);
    }

    #[test]
    fn test_kl_divergence_non_negative() {
        let p = vec![vec![0.5, 0.3, 0.2]];
        let q = vec![vec![0.1, 0.1, 0.8]];
        let kl = ternary_kl_divergence(&p, &q);
        assert!(kl >= 0.0, "KL divergence should be non-negative, got {}", kl);
    }

    #[test]
    fn test_kl_divergence_asymmetric() {
        let p = vec![vec![0.7, 0.2, 0.1]];
        let q = vec![vec![0.1, 0.3, 0.6]];
        let kl_pq = ternary_kl_divergence(&p, &q);
        let kl_qp = ternary_kl_divergence(&q, &p);
        assert!((kl_pq - kl_qp).abs() > 0.01, "KL should be asymmetric: pq={}, qp={}", kl_pq, kl_qp);
        assert!(kl_pq > 0.0);
        assert!(kl_qp > 0.0);
    }

    #[test]
    fn test_kl_divergence_single() {
        let p = [0.5, 0.25, 0.25];
        let q = [1.0/3.0; 3];
        let kl = ternary_kl_single(&p, &q);
        assert!(kl >= 0.0);
    }

    #[test]
    fn test_z3_distance_same() {
        assert_eq!(z3_distance(0, 0), 0.0);
        assert_eq!(z3_distance(1, 1), 0.0);
        assert_eq!(z3_distance(-1, -1), 0.0);
    }

    #[test]
    fn test_z3_distance_different() {
        assert_eq!(z3_distance(0, 1), 1.0);
        assert_eq!(z3_distance(0, -1), 1.0);
        assert_eq!(z3_distance(1, -1), 1.0);
    }

    #[test]
    fn test_is_ternary() {
        assert!(is_ternary(-1));
        assert!(is_ternary(0));
        assert!(is_ternary(1));
        assert!(!is_ternary(2));
        assert!(!is_ternary(-2));
    }

    #[test]
    fn test_z3_to_prob() {
        assert_eq!(z3_to_prob(-1), 0.0);
        assert_eq!(z3_to_prob(0), 0.5);
        assert_eq!(z3_to_prob(1), 1.0);
    }
}
