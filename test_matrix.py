import numpy as np
import heapq
import json

TEST_ITEM_NUM = 3260


def test_one_user(x):
    Ks = [5, 10, 20, 50, 100]  # matrix@k
    # Ks = [10]  # matrix@k
    rating = x[0]  # # user u's ratings for user u
    u = x[1]   #uid
    training_items = json.loads(x[2])
    user_pos_test = json.loads(x[3])
    if len(set(training_items+user_pos_test)) - len(set(training_items)) - len(set(user_pos_test)) != 0:
        print("wrong!!!!!!!!!")
        exit(-1)
    all_items = set(range(rating.shape[0]))

    test_items = list(all_items - set(training_items))
    r = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    return get_performance(user_pos_test, r, Ks)


def get_performance(user_pos_test, r, Ks):
    recall, ndcg = [], []
    for K in Ks:
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K))

    return {'recall': np.array(recall), 'ndcg': np.array(ndcg)}


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    r = r[:k]
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    # print(item_score)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    return r
