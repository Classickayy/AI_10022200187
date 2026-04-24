from src.retrieval import AdvancedRetriever


def test_keyword_score_fix_reduces_substring_noise():
    retriever = AdvancedRetriever()
    content = "This section explains fiscal policy and budget targets."
    query = "is in to"
    noisy = retriever._keyword_score(query, content, use_fix=False)
    fixed = retriever._keyword_score(query, content, use_fix=True)

    assert noisy >= fixed
