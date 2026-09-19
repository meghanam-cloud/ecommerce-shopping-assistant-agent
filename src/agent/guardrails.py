"""
Standalone guardrails evaluator. Runs BEFORE any LLM call.

Checks (each toggled in config.yaml -> guardrails):
  1. PII        - blocks input containing emails, phone numbers, card-like
                  number sequences typed by the user (we never want that
                  reaching a prompt or a log).
  2. Prompt injection - blocks common override phrases ("ignore previous
                  instructions", "you are now...", "reveal your system
                  prompt", etc).
  3. Topical scope   - the agent only answers product/catalog questions.
                  Off-topic questions (general chit-chat, unrelated domains)
                  are rejected before they burn an LLM call.
"""

import re
from dataclasses import dataclass

from src.utils.config_loader import settings

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\b(\+?\d{1,3}[-.\s]?)?\d{10}\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

_INJECTION_PATTERNS = [
    r"ignore (?:(?:all|the|any) )?(?:previous|prior|above) instructions",
    r"disregard (?:(?:all|the|any) )?(?:previous|prior|above) (?:instructions|rules)",
    r"you are now",
    r"act as (a|an) (?!ecommerce|shopping)",
    r"reveal (your|the) (system|hidden) prompt",
    r"print (your|the) (system|instructions)",
    r"developer mode",
    r"jailbreak",
    r"drop table",
    r"delete from",
    r"truncate table",
    r"update .* set ",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str = ""


def check_pii(text: str) -> GuardrailResult:
    if _EMAIL_RE.search(text) or _PHONE_RE.search(text) or _CARD_RE.search(text):
        return GuardrailResult(False, "Your message appears to contain personal information (email/phone/card number). Please rephrase without it.")
    return GuardrailResult(True)


def check_prompt_injection(text: str) -> GuardrailResult:
    if _INJECTION_RE.search(text):
        return GuardrailResult(False, "This request can't be processed - it looks like an attempt to override the assistant's instructions or modify data directly.")
    return GuardrailResult(True)


def check_topical_scope(text: str) -> GuardrailResult:
    keywords = settings.guardrails.allowed_topics_keywords
    if not keywords:
        return GuardrailResult(True)
    lowered = text.lower()
    if any(kw.lower() in lowered for kw in keywords):
        return GuardrailResult(True)
    return GuardrailResult(False, "I can only help with questions about our product catalog (pricing, ratings, sizes, delivery, categories, etc.). Could you rephrase your question around a product topic?")


def evaluate(text: str) -> GuardrailResult:
    """Run all enabled checks in order; return the first failure, else allow."""
    cfg = settings.guardrails

    if cfg.pii_check:
        result = check_pii(text)
        if not result.allowed:
            return result

    if cfg.prompt_injection_check:
        result = check_prompt_injection(text)
        if not result.allowed:
            return result

    if cfg.topical_scope_check:
        result = check_topical_scope(text)
        if not result.allowed:
            return result

    return GuardrailResult(True)
