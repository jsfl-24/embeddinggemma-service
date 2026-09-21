from embeddinggemma_service.cli import main


def test_cli_requires_text(capsys):
    assert main([]) == 2


def test_cli_embeds_text(capsys):
    assert main(["Hello world"]) == 0
    out = capsys.readouterr().out
    assert "Model: embeddinggemma-300m" in out
    assert "Dimension: 768" in out
    assert "Embedding:" in out