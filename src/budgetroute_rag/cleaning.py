import re


def remove_rst_roles(text: str) -> str:
    """
    Remove RST inline role markup while preserving its content.

    Examples:
        :class:`KMeans` -> KMeans
        :func:`fit` -> fit
        :math:`x^2` -> x^2
    """

    pattern = r":([a-zA-Z][\w.-]*):`([^`]+)`"

    def replace_role(match):
        role_name = match.group(1)
        content = match.group(2).strip()

        # If a role contains visible text and a reference target,
        # keep the visible text.
        if "<" in content and content.endswith(">"):
            content = content.split("<", 1)[0].strip()

        # Remove Sphinx's shortened-display marker.
        content = content.lstrip("~")

        return content

    return re.sub(pattern, replace_role, text, flags=re.DOTALL)
def normalize_whitespace(text: str) -> str:
    """
    Normalize unnecessary whitespace while preserving code-block indentation.
    """

    lines = text.splitlines()
    cleaned_lines = []

    in_code_block = False
    code_indent = None

    for line in lines:

        # Detect the beginning of an RST code block.
        if re.match(r"^\s*\.\.\s+(code-block|sourcecode)::", line):
            in_code_block = True
            code_indent = None
            cleaned_lines.append(line.strip())
            continue

        # Preserve blank lines inside code blocks.
        if in_code_block and not line.strip():
            cleaned_lines.append("")
            continue

        # Preserve code indentation.
        if in_code_block:
            indentation = len(line) - len(line.lstrip())

            if indentation > 0:
                if code_indent is None:
                    code_indent = indentation

                if indentation >= code_indent:
                    cleaned_lines.append(line)
                    continue

            # The code block has ended.
            in_code_block = False
            code_indent = None

        # Normalize ordinary text.
        normalized = re.sub(r"[ \t]+", " ", line.strip())

        cleaned_lines.append(normalized)

    # Remove unnecessary blank lines at the beginning and end.
    result = "\n".join(cleaned_lines).strip()

    # Reduce three or more consecutive newlines to two.
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result
def clean_rst_text(text: str) -> str:
    """
    Apply all text-cleaning operations in sequence.
    """

    text = remove_rst_roles(text)
    text = normalize_whitespace(text)

    return text