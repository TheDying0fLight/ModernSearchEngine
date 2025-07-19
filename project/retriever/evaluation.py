import numpy as np
import math


def compute_metrics(eval_pred: tuple):
    logits, _ = eval_pred
    # Convert logits to similarity scores in [0,1]
    conf: np.ndarray = (logits + 1) / 2
    N, M = conf.shape
    metrics = {}

    # Compute true ranks for recall@k
    top_idxs = conf.argsort(axis=1)[:, ::-1]
    true_idxs = np.tile(np.arange(M), math.ceil(N / M))[:N][:, None]
    ranks_pos: np.ndarray = np.argmax(top_idxs == true_idxs, axis=1)

    # Determine ks (up to 2^6 if N large)
    max_pow = int(np.log2(N))
    ks = [2**i for i in range(min(max_pow + 1, 6))]
    metrics = {f"recall@{k}": np.mean(ranks_pos < k) for k in ks}
    metrics['mean_rank'] = ranks_pos.mean()
    metrics['median_rank'] = np.median(ranks_pos)
    metrics['mean_rank_norm'] = ranks_pos.mean() / M
    metrics['median_rank_norm'] = np.median(ranks_pos) / M
    metrics['min_rank'] = ranks_pos.max()
    metrics['min_rank_norm'] = ranks_pos.max() / M

    for p in [1, 2, 5, 10, 25, 50]:
        metrics[f"recall@{p}%"] = np.mean(ranks_pos < max(min(N, M) * p / 100, 1))

    # Flatten positive (diagonal) and negative (off-diagonal) scores
    mask = np.tile(np.eye(M, dtype=bool), math.ceil(N / M)).T[:N]
    conf_pos = conf[mask]
    conf_neg = conf[~mask]
    conf_neg = conf_neg[conf_neg >= 0]
    y_true = np.concatenate([np.ones_like(conf_pos), np.zeros_like(conf_neg)])
    y_scores = np.concatenate([conf_pos, conf_neg])

    # Compute recall at decile thresholds
    total_pos = y_true.sum()
    total_neg = y_true.size - total_pos
    order = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[order]
    cum_pos = np.cumsum(y_true_sorted)
    cum_neg = np.cumsum(1 - y_true_sorted)

    # Best threshold maximizing TPR + TNR
    tpr = cum_pos / total_pos
    tnr = (total_neg - cum_neg) / total_neg
    score = tpr + tnr
    best_idx = np.argmax(score)
    best_thresh = float(y_scores[order][best_idx])
    best_score = float(score[best_idx])
    metrics['best_threshold'] = best_thresh
    metrics['best_score'] = best_score

    return metrics
