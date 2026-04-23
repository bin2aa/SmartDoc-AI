"""Utility helpers for SmartDoc AI."""

import re
from typing import List, Tuple

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Regex: @filename — captures letters, digits, dots, hyphens, underscores
_MENTION_PATTERN = re.compile(r"@([\w.\-]+)")


def parse_file_mentions(raw_query: str) -> Tuple[str, List[str]]:
    """
    Extract @filename mentions from a user query.

    Scans the raw query for tokens starting with ``@`` (e.g. ``@vd1.pdf``,
    ``@report-2024.docx``).  The ``@`` symbol is removed from the returned
    clean query so the LLM can read it naturally.  Extra whitespace is
    collapsed.

    Args:
        raw_query: The original user input, potentially containing
            ``@filename`` mentions.

    Returns:
        A tuple of ``(clean_query, mentioned_files)`` where
        * *clean_query* is the query with ``@`` prefixes removed and
          whitespace normalised.
        * *mentioned_files* is a list of filenames (without ``@``) found in
          the query.  Duplicates are preserved in order of first appearance.

    Example::

        >>> parse_file_mentions("So sánh @vd1.pdf và @vd2.docx về input")
        ('So sánh vd1.pdf và vd2.docx về input', ['vd1.pdf', 'vd2.docx'])
    """
    if not raw_query:
        return raw_query, []

    # Find all @mentions
    mentions = _MENTION_PATTERN.findall(raw_query)

    if not mentions:
        return raw_query.strip(), []

    # Remove the @ prefix from the query for each mention
    clean = raw_query
    for mention in mentions:
        clean = clean.replace(f"@{mention}", mention)

    # Collapse extra whitespace
    clean = re.sub(r"\s+", " ", clean).strip()

    logger.info(f"Parsed @mentions: {mentions} from query")

    return clean, mentions