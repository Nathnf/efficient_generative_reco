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


def get_topk_results(predictions, scores, targets, k, all_items=None):
    results = []
    B = len(targets)
    # predictions = [_.split("Response:")[-1] for _ in predictions]
    predictions = [_.strip().replace(" ", "") for _ in predictions]
    # print(predictions)##################
    incorrect_pred_no, correct_pred_no = 0, 0

    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                incorrect_pred_no += 1
                scores[i] = -1000

            else:
                correct_pred_no += 1

    logger.debug(
        "Total incorrect predictions: {}, Total correct predictions: {}".format(incorrect_pred_no, correct_pred_no)
    )
    logger.info("Ratio of correct predictions: {:.4f}".format(correct_pred_no / len(predictions)))

    # # To get the ratio of correct predictions per codebook level
    # n_query = 4
    # if n_query is not None:
    #     position_correct_counts = [0] * n_query
    #     position_total_counts = [0] * n_query

    # print(scores)
    for b in range(B):
        batch_seqs = predictions[b * k : (b + 1) * k]
        batch_scores = scores[b * k : (b + 1) * k]

        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        # print(pairs)
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        target_item = targets[b]
        one_results = []

        for sorted_pred in sorted_pairs:
            if sorted_pred[0] == target_item:
                one_results.append(1)
            else:
                one_results.append(0)

            # # To get the ratio of correct predictions per codebook level - 2
            # if n_query is not None:
            #     for i in range(n_query):
            #         pred_tokens_list = parse_item(sorted_pred[0])
            #         target_tokens_list = parse_item(target_item)
            #         if pred_tokens_list[i] == target_tokens_list[i]:
            #             position_correct_counts[i] += 1
            #         position_total_counts[i] += 1

        results.append(one_results)

    # # To get the ratio of correct predictions per codebook level - 3
    # if n_query is not None:
    #     position_accuracies = [
    #         position_correct_counts[i] / position_total_counts[i] if position_total_counts[i] > 0 else 0
    #         for i in range(n_query)
    #     ]
    #     for i, acc in enumerate(position_accuracies):
    #         logger.debug("Position {}: {:.4f}".format(i, acc))

    return results


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
