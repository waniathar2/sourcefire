from src.chain.prompts import assemble_prompt, estimate_tokens, truncate_chunks


def test_estimate_tokens():
    text = "hello world this is a test"
    tokens = estimate_tokens(text)
    assert 4 <= tokens <= 10


def test_truncate_chunks_within_budget():
    chunks = [
        {"code": "x " * 100, "filename": "a.dart", "relevance": 0.9, "priority": "direct"},
        {"code": "y " * 100, "filename": "b.dart", "relevance": 0.7, "priority": "semantic"},
        {"code": "z " * 100, "filename": "c.dart", "relevance": 0.5, "priority": "graph"},
    ]
    result = truncate_chunks(chunks, max_tokens=500)
    assert len(result) <= 3
    assert result[0]["filename"] == "a.dart"


def test_truncate_chunks_drops_graph_first():
    chunks = [
        {"code": "x " * 500, "filename": "a.dart", "relevance": 0.9, "priority": "direct"},
        {"code": "y " * 500, "filename": "b.dart", "relevance": 0.7, "priority": "semantic"},
        {"code": "z " * 500, "filename": "c.dart", "relevance": 0.5, "priority": "graph"},
    ]
    result = truncate_chunks(chunks, max_tokens=400)
    filenames = [c["filename"] for c in result]
    assert "c.dart" not in filenames or "a.dart" in filenames


def test_assemble_prompt_contains_sections():
    prompt = assemble_prompt(
        mode="debug",
        query="type Null error in auth",
        chunks=[{"code": "class Auth {}", "filename": "auth.dart", "location": "1:0", "relevance": 0.9}],
        claude_md="# Rules",
        memory_content="## Conventions",
        history=[],
        model="gemini-2.5-flash",
    )
    assert "Cravv Observatory" in prompt["system"]
    assert "# Rules" in prompt["system"]
    assert "auth.dart" in prompt["context"]
    assert prompt["query"] == "type Null error in auth"
