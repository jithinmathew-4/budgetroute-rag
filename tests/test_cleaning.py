from budgetroute_rag.cleaning import (
    remove_rst_roles,
    normalize_whitespace,
    clean_rst_text,
)


def test_removes_class_role():
    text = "Use :class:`KMeans` for clustering."

    result = remove_rst_roles(text)

    assert result == "Use KMeans for clustering."


def test_removes_function_role():
    text = "Call :func:`fit` to train the model."

    result = remove_rst_roles(text)

    assert result == "Call fit to train the model."


def test_preserves_math_content():
    text = "The value is :math:`x^2`."

    result = remove_rst_roles(text)

    assert result == "The value is x^2."


def test_preserves_reference_display_text():
    text = ":ref:`K-Means <k_means>`"

    result = remove_rst_roles(text)

    assert result == "K-Means"


def test_preserves_qualified_class_name():
    text = ":class:`~sklearn.cluster.KMeans`"

    result = remove_rst_roles(text)

    assert result == "sklearn.cluster.KMeans"
    from budgetroute_rag.cleaning import normalize_whitespace


def test_normalizes_repeated_spaces():
    text = "KMeans     is an algorithm."

    result = normalize_whitespace(text)

    assert result == "KMeans is an algorithm."


def test_removes_unnecessary_blank_lines():
    text = "First paragraph.\n\n\n\nSecond paragraph."

    result = normalize_whitespace(text)

    assert result == "First paragraph.\n\nSecond paragraph."


def test_preserves_code_block_indentation():
    text = (
        ".. code-block:: python\n"
        "\n"
        "    def greet():\n"
        "        print('Hello')\n"
        "\n"
        "This is normal text."
    )

    result = normalize_whitespace(text)

    assert "    def greet():" in result
    assert "        print('Hello')" in result


def test_trims_leading_and_trailing_whitespace():
    text = "   KMeans is useful.   "

    result = normalize_whitespace(text)

    assert result == "KMeans is useful."
def test_clean_rst_text_combines_cleaning_steps():
    text = (
        "The   :class:`KMeans` algorithm.\n\n\n"
        "Use :func:`fit` to train it."
    )

    result = clean_rst_text(text)

    expected = (
        "The KMeans algorithm.\n\n"
        "Use fit to train it."
    )

    assert result == expected