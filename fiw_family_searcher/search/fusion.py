# fiw_family_searcher/search/fusion.py
from collections import defaultdict


def reciprocal_rank_fusion(ranked_lists_with_scores, k=60):
    """
    Performs Reciprocal Rank Fusion.
    Each list in ranked_lists_with_scores is a list of (item_id, score) tuples.
    Scores are assumed to be similarity scores (higher is better).
    If scores are distances (lower is better), they need to be converted first.
    """
    rrf_scores = defaultdict(float)

    for ranked_list in ranked_lists_with_scores:
        # Sort by score in descending order (higher score = better rank)
        # If scores are distances, sort ascending and then convert distance to score for RRF formula if needed,
        # or just use rank. The original RRF uses ranks.

        # Assuming scores are similarities (higher is better)
        # If using distances from FAISS L2, convert them: score = 1 / (1 + distance)
        # For simplicity here, let's assume items are already ranked and we only care about rank.
        # Or, if scores are already similarities:

        # Let's make it flexible: if scores are provided, sort by them. If not, assume items are already ranked.

        # This implementation will sort by score (desc) to get rank
        # If items are already ranked and no scores, then rank = index + 1

        # Let's assume ranked_lists_with_scores contains [(item_id, similarity_score), ...]
        # Sort by similarity_score descending to get rank
        sorted_list = sorted(ranked_list, key=lambda x: x[1], reverse=True)

        for rank, (item_id, _) in enumerate(sorted_list):
            rrf_scores[item_id] += 1.0 / (k + rank + 1)  # rank is 0-indexed

    # Sort items by their RRF score in descending order
    fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused_results  # List of (item_id, rrf_score)