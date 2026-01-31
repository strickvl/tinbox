import re


def extract_whitespace_formatting(content: str) -> tuple[str, str, str]:
    """Extract prefix, core content, and suffix from text.

    Args:
        content: Input text content

    Returns:
        tuple: (prefix_whitespace, core_content, suffix_whitespace)
    """
    if not isinstance(content, str):
        return "", content, ""

    core = content.strip()
    
    if not core:
        # If content is only whitespace, treat it all as prefix
        return content, "", ""

    prefix_match = re.match(r'^(\s*)', content)
    suffix_match = re.search(r'(\s*)$', content)

    prefix = prefix_match.group(1) if prefix_match else ""
    suffix = suffix_match.group(1) if suffix_match else ""

    return prefix, core, suffix