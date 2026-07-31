from actapint.plotting import normalize_svg


def test_normalize_svg_is_responsive_accessible_and_idempotent(tmp_path):
    svg = tmp_path / "plot.svg"
    svg.write_text(
        '<svg width="72pt" height="36pt" viewBox="0 0 72 36" '
        'xmlns="http://www.w3.org/2000/svg"><path d="M0 0"/></svg>',
        encoding="utf-8",
    )

    for _ in range(2):
        normalize_svg(
            svg,
            title="Heat & transport",
            description="A < B.",
        )

    result = svg.read_text(encoding="utf-8")
    assert 'width="72"' in result
    assert 'height="36"' in result
    assert 'role="img"' in result
    assert 'aria-labelledby="figure-title figure-description"' in result
    assert result.count('id="figure-title"') == 1
    assert result.count('id="figure-description"') == 1
    assert "Heat &amp; transport" in result
    assert "A &lt; B." in result
