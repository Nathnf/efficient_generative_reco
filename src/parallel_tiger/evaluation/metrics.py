import math
import logging
from parallel_tiger.generation.vectorized_constraints import parse_item

logger = logging.getLogger(__name__)


def get_eval_metrics_results(predictions, labels):

    # predictions = [_.strip().replace(" ","") for _ in predictions]
    # labels = [_.strip().replace(" ","") for _ in labels]

    predictions = [str(pred[1:5]) for pred in predictions]
    labels = [str(label[:4]) for label in labels]

    results = []

    for i in range(len(labels)):
        pred = predictions[i]
        label = labels[i]

        one_results = []

        if pred == label:
            one_results.append(1)
        else:
            one_results.append(0)

        results.append(one_results)

    metrics_results = get_metrics_results(results, metrics=["hit@1"])

    metric = dict()
    for k, v in metrics_results.items():
        metric[k.replace("@", "_at_")] = v / len(labels)

    return metric


def get_topk_results(predictions, scores, targets, k, all_items, filter_invalid=True, per_level_stats=False):
    results = []
    B = len(targets)
    predictions = [_.strip().replace(" ", "") for _ in predictions]
    incorrect_pred_no, correct_pred_no = 0, 0

    for i, seq in enumerate(predictions):
        if seq not in all_items:
            # if invalid_count < 10:
            #     print(f"Warning: {seq} not in all_items, setting score to -1000 (initially {scores[i]})")
            # invalid_count += 1
            incorrect_pred_no += 1
            if filter_invalid:
                scores[i] = -1000
        else:
            correct_pred_no += 1

    # To get the ratio of correct predictions per codebook level
    if per_level_stats:
        n_query = 4
        # position_correct_counts = [0] * n_query
        # position_total_counts = [0] * n_query
        position_correct_counts = [0] * n_query
        position_total_counts = [B] * n_query  # one per example
        subseq_correct_counts = [0] * n_query
        subseq_total_counts = [0] * n_query

    # print(scores)
    for b in range(B):
        batch_seqs = predictions[b * k : (b + 1) * k]
        batch_scores = scores[b * k : (b + 1) * k]

        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        # print(pairs)
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        target_item = targets[b]
        target_tokens = parse_item(target_item)
        candidate_tokens = [parse_item(seq) for seq, _ in sorted_pairs]

        one_results = [1 if seq == target_item else 0 for seq, _ in sorted_pairs]
        results.append(one_results)

        if per_level_stats:
            # # --- per-level accuracy ---
            # for i in range(n_query):
            #     for pred_tokens in candidate_tokens:
            #         if pred_tokens[i] == target_tokens[i]:
            #             position_correct_counts[i] += 1
            #         position_total_counts[i] += 1

            # Collect tokens per level across all k candidates
            tokens_per_level = [set() for _ in range(n_query)]
            target_tokens = parse_item(targets[b])
            for candidate in batch_seqs:
                cand_tokens = parse_item(candidate)
                for i in range(n_query):
                    if cand_tokens[i] == target_tokens[i]:
                        tokens_per_level[i].add(candidate)
            
            # If at least one candidate is correct at level i, count it
            for i in range(n_query):
                if len(tokens_per_level[i]) > 0:
                    position_correct_counts[i] += 1

            # --- per-subsequence (prefix) accuracy ---
            for L in range(1, n_query + 1):
                target_prefix = tuple(target_tokens[:L])
                if any(tuple(pred[:L]) == target_prefix for pred in candidate_tokens):
                    subseq_correct_counts[L-1] += 1
                subseq_total_counts[L-1] += 1


    if per_level_stats:
        position_accuracies = [
            position_correct_counts[i] / position_total_counts[i] if position_total_counts[i] > 0 else 0
            for i in range(n_query)
        ]
        subseq_accuracies = [
            subseq_correct_counts[i] / subseq_total_counts[i] if subseq_total_counts[i] > 0 else 0
            for i in range(n_query)
        ]

        for i, acc in enumerate(position_accuracies):
            logger.debug(f"Per-level accuracy token {i+1}: {acc:.4f}")
        
        for i, acc in enumerate(subseq_accuracies):
            logger.debug(f"Per-subsequence accuracy up to token {i+1}: {acc:.4f}")

    return results, correct_pred_no, incorrect_pred_no


def get_metrics_results(topk_results, metrics):
    res = {}
    for m in metrics:
        if m.lower().startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        else:
            raise NotImplementedError

    return res


def ndcg_k(topk_results, k):
    """
    Since we apply leave-one-out, each user only have one ground truth item, so the idcg would be 1.0
    """
    ndcg = 0.0
    for row in topk_results:
        res = row[:k]
        one_ndcg = 0.0
        for i in range(len(res)):
            one_ndcg += res[i] / math.log(i + 2, 2)
        ndcg += one_ndcg
    return ndcg


def hit_k(topk_results, k):
    hit = 0.0
    for row in topk_results:
        res = row[:k]
        if sum(res) > 0:
            hit += 1
    return hit
