from src.data_engineering import DataProcessor


def test_chunking_strategies_generate_chunks():
    dp = DataProcessor()
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    fixed = dp.fixed_word_chunking(text, chunk_size=4, overlap=1)
    recursive = dp.recursive_chunking(text, chunk_size=4, overlap=1)

    assert len(fixed) > 0
    assert len(recursive) > 0


def test_chunking_comparison_output_shape():
    dp = DataProcessor()
    report = dp.compare_chunking_strategies()

    assert "chunk_params" in report
    assert "fixed" in report
    assert "recursive" in report
