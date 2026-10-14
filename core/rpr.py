"""RPR: reciprocal precedent reranking and label-constrained reselection."""
from collections import defaultdict
from .data import charge_labels, article_labels


def rerank(facet_lists, alpha=60, scope="intersection"):
    """Sum 1 / (one-based rank + alpha); absent facets contribute zero.

    Intersection follows the PDF definition: candidates appear in all three
    lists. Union is retained as an explicitly selected historical variant.
    """
    if alpha < 0 or scope not in ("union", "intersection"):
        raise ValueError("Expected nonnegative alpha and union/intersection scope")
    scores = defaultdict(float)
    memberships = []
    for ranked in facet_lists:
        valid = list(dict.fromkeys(int(i) for i in ranked if int(i) >= 0))
        memberships.append(set(valid))
        for rank, case_id in enumerate(valid, 1):
            scores[case_id] += 1 / (rank + alpha)
    if scope == "intersection":
        common = set.intersection(*memberships) if memberships else set()
        scores = {key: value for key, value in scores.items() if key in common}
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def select_second_round(ranked, precedents, charge, law_article, k):
    """Keep top-K matching labels, then backfill from first-round precedents."""
    charge_set, article_set = set(charge_labels(charge)), set(article_labels(law_article))
    selected = [i for i, _ in ranked
                if set(charge_labels(precedents[i]["labels"]["charge"])) == charge_set
                and set(article_labels(precedents[i]["labels"]["law_article"])) == article_set][:k]
    for case_id, _ in ranked[:k]:
        if len(selected) == k:
            break
        if case_id not in selected:
            selected.append(case_id)
    return selected
