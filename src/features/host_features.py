"""Host-only lexical features computed from the PhiUSIIL `Domain` field.

Every legitimate PhiUSIIL URL is a bare `https://www.<host>` homepage, while phishing URLs
often carry paths, queries, `http`, or no `www.` prefix. Features derived from the path,
scheme, or `www.` prefix would therefore learn the dataset's collection process. These
features use only the host with a leading `www.` removed, and need no page fetch.
"""

from __future__ import annotations

import ipaddress
import math
from collections import Counter

CANDIDATE_FEATURES = (
    "host_length",
    "host_digit_ratio",
    "host_hyphen_count",
    "host_label_count",
    "host_entropy",
    "longest_label_length",
    "tld_length",
    "host_is_ip",
)


def normalize_host(domain: str) -> str:
    """Lowercase, drop a trailing dot and port, and remove one leading `www.`."""
    host = domain.strip().lower().rstrip(".")
    if host.startswith("[") and "]" in host:  # bracketed IPv6, optionally with a port
        host = host[1 : host.index("]")]
    elif host.count(":") == 1:
        host = host.split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    return -sum(count / total * math.log2(count / total) for count in Counter(text).values())


def host_features(host: str) -> dict[str, float]:
    """Return every candidate feature for an already normalized, non-empty host."""
    if not host:
        raise ValueError("Host must be non-empty.")
    labels = host.split(".")
    ip = is_ip(host)
    return {
        "host_length": float(len(host)),
        "host_digit_ratio": sum(char.isdigit() for char in host) / len(host),
        "host_hyphen_count": float(host.count("-")),
        "host_label_count": float(len(labels)),
        "host_entropy": shannon_entropy(host),
        "longest_label_length": float(max(len(label) for label in labels)),
        "tld_length": 0.0 if ip else float(len(labels[-1])),
        "host_is_ip": 1.0 if ip else 0.0,
    }
