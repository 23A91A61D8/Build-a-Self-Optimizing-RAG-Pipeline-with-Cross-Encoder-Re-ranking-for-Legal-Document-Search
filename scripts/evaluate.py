import json
import requests
import math

BASELINE_URL = "http://127.0.0.1:9000/api/v1/retrieve/baseline"
RERANKED_URL = "http://127.0.0.1:9000/api/v1/retrieve/reranked"

def reciprocal_rank(results, relevant_docs):
    for i, item in enumerate(results):
        if item["doc_id"] in relevant_docs:
            return 1 / (i + 1)
    return 0

def dcg(results, relevant_docs, k=10):
    score = 0
    for i, item in enumerate(results[:k]):
        if item["doc_id"] in relevant_docs:
            score += 1 / math.log2(i + 2)
    return score

def evaluate():
    with open("../evaluation/queries.json", "r") as f:
        queries = json.load(f)

    baseline_rr = []
    reranked_rr = []

    baseline_dcg = []
    reranked_dcg = []

    for q in queries:
        query = q["query_text"]
        relevant = q["relevant_docs"]

        # Baseline
        b_res = requests.get(BASELINE_URL, params={"query": query, "k": 10}).json()["results"]
        r_res = requests.get(RERANKED_URL, params={"query": query, "k": 10}).json()["results"]

        baseline_rr.append(reciprocal_rank(b_res, relevant))
        reranked_rr.append(reciprocal_rank(r_res, relevant))

        baseline_dcg.append(dcg(b_res, relevant))
        reranked_dcg.append(dcg(r_res, relevant))

    results = {
        "baseline": {
            "mrr_at_5": sum(baseline_rr) / len(baseline_rr),
            "ndcg_at_10": sum(baseline_dcg) / len(baseline_dcg)
        },
        "reranked": {
            "mrr_at_5": sum(reranked_rr) / len(reranked_rr),
            "ndcg_at_10": sum(reranked_dcg) / len(reranked_dcg)
        }
    }

    with open("../results/evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Evaluation Completed ✅")
    print(results)

if __name__ == "__main__":
    evaluate()