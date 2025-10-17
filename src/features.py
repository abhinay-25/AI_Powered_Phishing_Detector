import re
import tldextract
from typing import Dict

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "update", "secure", "account", "bank", "confirm",
    "urgent", "password", "invoice", "limited", "alert", "suspend",
]


def count_occurrences(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text or "", flags=re.IGNORECASE))


def url_features(url: str) -> Dict[str, float]:
    url = url or ""
    ext = tldextract.extract(url)
    domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain

    feats = {
        "url_length": len(url),
        "has_at_symbol": int("@" in url),
        "has_hyphen_domain": int("-" in (ext.domain or "")),
        "is_ip_address": int(bool(re.fullmatch(r"(\d{1,3}\.){3}\d{1,3}", ext.domain or ""))),
        "num_digits": sum(c.isdigit() for c in url),
        "num_subdirs": url.count("/"),
        "num_params": url.count("="),
        "https": int(url.lower().startswith("https")),
        "num_suspicious_keywords": sum(k in url.lower() for k in SUSPICIOUS_KEYWORDS),
        "domain_length": len(domain or ""),
    }
    return feats


def email_features(subject: str, body: str) -> Dict[str, float]:
    subject = subject or ""
    body = body or ""
    text = f"{subject}\n{body}".lower()

    feats = {
        "subject_length": len(subject),
        "body_length": len(body),
        "num_links": count_occurrences(body, r"https?://"),
        "num_exclamations": body.count("!"),
        "num_dollar": body.count("$"),
        "urgent_keywords": sum(k in text for k in ["urgent", "immediately", "asap", "important", "action required"]),
        "num_suspicious_keywords": sum(k in text for k in SUSPICIOUS_KEYWORDS),
    }
    return feats
