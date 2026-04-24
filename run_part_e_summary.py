import json
from pathlib import Path

from src.rag_pipeline import RAGPipeline
from src.evaluator import AdversarialTester


def main():
    pipeline = RAGPipeline()
    tester = AdversarialTester(pipeline)
    report = tester.run_evaluation()

    print("\n=== PART E SUMMARY ===")
    print(f"Queries tested: {report.get('queries_tested', 0)}")

    agg = report.get("aggregate_metrics", {})
    rag = agg.get("rag", {})
    pure = agg.get("pure_llm", {})

    print("\nRAG Metrics")
    print(f"- Accuracy rate: {rag.get('accuracy_rate', 0)}")
    print(f"- Hallucination rate: {rag.get('hallucination_rate', 0)}")
    print(f"- Avg response consistency: {rag.get('avg_response_consistency', 0)}")
    print(f"- Valid samples: {rag.get('valid_samples', 0)}")

    print("\nPure LLM Metrics")
    print(f"- Accuracy rate: {pure.get('accuracy_rate', 0)}")
    print(f"- Hallucination rate: {pure.get('hallucination_rate', 0)}")
    print(f"- Avg response consistency: {pure.get('avg_response_consistency', 0)}")
    print(f"- Valid samples: {pure.get('valid_samples', 0)}")

    report_path = Path("logs") / "adversarial_evaluation.json"
    print(f"\nDetailed report saved to: {report_path}")

    # Optional small preview for quick checking
    per_query = report.get("per_query_results", [])
    if per_query:
        print("\nPer-query quick preview:")
        for item in per_query:
            q = item["query"]["query"]
            rag_eval = item["rag"]["evaluation"]
            pure_eval = item["pure_llm"]["evaluation"]
            print(f"- Query: {q}")
            print(
                f"  RAG(acc={rag_eval.get('accuracy_hit')}, hall={rag_eval.get('hallucination_detected')}, cons={rag_eval.get('response_consistency')})"
            )
            print(
                f"  LLM(acc={pure_eval.get('accuracy_hit')}, hall={pure_eval.get('hallucination_detected')}, cons={pure_eval.get('response_consistency')})"
            )


if __name__ == "__main__":
    main()
