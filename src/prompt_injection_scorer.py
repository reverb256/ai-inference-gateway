"""
Tiered prompt injection detection scorer.

Provides a layered detection pipeline:
  Layer 1 — regex pattern matching (fast, deterministic)
  Layer 2 — heuristic analysis (instruction density, role confusion, etc.)

Produces an InjectionRisk with a continuous score (0.0–1.0) and a
classification level: clean / suspicious / likely / confirmed.
"""

import re;
import time;
from dataclasses import dataclass, field;
from typing import Tuple;


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class InjectionRisk:
    """Result of prompt-injection scoring."""

    score: float = 0.0;
    level: str = "clean";
    triggers: list[str] = field(default_factory=list);
    heuristic_flags: list[str] = field(default_factory=list);
    latency_ms: float = 0.0;


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class PromptInjectionScorer:

    # ---- raw pattern strings (compiled in __init__) ----------------------
    #
    # Base patterns are copied from security_filter.py INJECTION_PATTERNS
    # and extended with additional categories.

    _PATTERN_STRINGS: list[str] = [
        # --- Instruction override ---
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"disregard\s+(everything\s+)?(above|before|prior)",
        r"forget\s+(the\s+)?(above|previous|prior|everything)",
        r"override\s+(your\s+)?instructions",
        r"pretend\s+you\s+are\s+not",
        r"act\s+as\s+if\s+you\s+are",
        r"you\s+are\s+now\s+(a\s+)?(DAN|evil|unfiltered|uncensored)",
        r"stop\s+being\s+(an?\s+)?(AI|assistant|helpful)",
        r"new\s+instructions?\s*:",
        r"system\s*(override|update|reset|instruction)\s*:",
        # --- Role play / persona ---
        r"simulate\s+(being|a|an)\s+(?!a\s+professional)",
        r"role[\-\s]?play\s+as",
        r"pretend\s+to\s+be\s+(?!a\s+professional)",
        r"you\s+are\s+no\s+longer\s+(an?\s+)?(AI|assistant|LLM)",
        r"(always|never)\s+(respond|answer|reply|comply|refuse)",
        # --- System prompt extraction ---
        r"(what\s+are\s+your|show\s+me\s+your|reveal\s+your)\s+(system|initial|original)\s+(prompt|instructions)",
        r"repeat\s+(your|the)\s+(system|initial|original)\s+(prompt|instructions)",
        r"output\s+(your|the)\s+system\s+prompt",
        # --- Credential/secret requests ---
        r"(show|reveal|give|send|share|display|print)\s+(me\s+)?(your\s+)?(api\s+key|secret|password|token|private\s+key|seed|credential)",
        r"(cat|type|read|print|echo)\s+.*(\.env|\.ssh|\.config|/etc/passwd|/etc/shadow)",
        r"\$\(\s*cat\s+",
        r"\b(exec|eval|system|subprocess|os\.system|os\.popen)\s*\(",
        # --- Hidden HTML/invisible content ---
        r"<!--\s*(system|important|instruction|override|admin)",
        r"<style[^>]*>[\s\S]*?(position\s*:\s*absolute|visibility\s*:\s*hidden|display\s*:\s*none|opacity\s*:\s*0|font-size\s*:\s*0|color\s*:\s*transparent)",
        r"<img\s+[^>]*(?:alt|title|src)\s*=\s*[\"'][^\"']*(?:ignore|override|system|instruction)[^\"']*[\"']",
        r"data:text/html[,;].*(?:ignore|override|system)",
        # --- Encoded payloads ---
        r"(base64|b64|atob|btoa)\s*[\(\[]",
        r"\\x[0-9a-fA-F]{2}.*\\x[0-9a-fA-F]{2}",
        r"\\u[0-9a-fA-F]{4}.*\\u[0-9a-fA-F]{4}",
        r"&#x[0-9a-fA-F]+;",
        # --- Shell injection in content ---
        r"(curl|wget|bash|sh|python|perl|ruby|node)\s+.*\|.*\|",
        r"(curl|wget)\s+https?://\S+\s*\|\s*(bash|sh|python)",
        r"`[^`]*(?:rm|curl|wget|bash|sh|chmod|chown)[^`]*`",
        # --- Multi-language (Korean, Chinese, Japanese, Russian) ---
        r"이전\s*(모든\s*)?지시",
        r"무시하고",
        r"API\s*키\s*(보여|알려|전달)",
        r"忽略\s*(之前|所有|上述)\s*(指令|指示)",
        r"假装\s*(你|您)\s*(不是|是)",
        r"前\s*の\s*指示\s*を\s*無視",
        r"игнорир(уй|овать)\s*(все\s+)?предыдущ",
        r"покажи\s+(мне\s+)?(свой\s+)?(api|ключ|пароль)",
        # --- Token manipulation (chat template injection) ---
        r"<\|im_start\|>",
        r"\[INST\]",
        r"###\s*Instruction",
        r"<<SYS>>",
        r"<\|endoftext\|>",
        r"<\|system\|>",
        # --- Additional jailbreak keywords ---
        r"\bjailbreak\b",
        r"\bdan\b\s+\d+\.\d+",
        r"\bunjailbreak\b",
        r"\bdeveloper\s+mode\b",
        r"\b(totally|completely)\s+unfiltered\b",
        # --- Additional role manipulation ---
        r"from\s+now\s+on\s+you\s+are",
        r"you\s+have\s+been\s+freed",
        r"no\s+(moral|ethical|safety)\s+(restrictions?|guidelines?|limits?|constraints?)",
        # --- System prompt leakage attempts (heuristic-level patterns) ---
        r"what\s+(were|are)\s+the\s+instructions",
        r"print\s+your\s+(system|hidden)\s+(prompt|message)",
        r"reveal\s+everything\s+above",
        r"dump\s+your\s+(system\s+)?prompt",
    ];

    # Imperative verbs used for instruction-density heuristic
    _IMPERATIVE_VERBS: set[str] = {
        "ignore", "forget", "disregard", "override", "tell", "show",
        "reveal", "print", "output", "display", "say", "respond",
        "answer", "reply", "repeat", "echo", "dump", "extract",
        "pretend", "act", "simulate", "role-play", "roleplay",
        "stop", "never", "always", "must", "shall", "will",
        "provide", "give", "send", "share", "list", "name",
        "translate", "convert", "encode", "decode", "write",
        "read", "cat", "exec", "eval", "run", "execute",
    };

    # ---- constructor -----------------------------------------------------

    def __init__(self) -> None:
        self.REGEX_PATTERNS: list[re.Pattern] = [
            re.compile(p, re.IGNORECASE) for p in self._PATTERN_STRINGS
        ];

    # ---- public API ------------------------------------------------------

    async def score(self, text: str) -> InjectionRisk:
        """Full scoring pipeline: regex + heuristics -> InjectionRisk."""

        t0 = time.perf_counter();

        regex_score, triggers = self._regex_scan(text);
        heur_score, flags = self._heuristic_analysis(text);

        combined = min(1.0, regex_score + heur_score);

        t1 = time.perf_counter();

        return InjectionRisk(
            score=round(combined, 4),
            level=self._classify(combined),
            triggers=triggers,
            heuristic_flags=flags,
            latency_ms=round((t1 - t0) * 1000.0, 3),
        );

    # ---- Layer 1: regex --------------------------------------------------

    def _regex_scan(self, text: str) -> Tuple[float, list[str]]:
        """Return (score, matched_pattern_strings)."""
        hits: list[str] = [];
        for pat in self.REGEX_PATTERNS:
            if pat.search(text):
                hits.append(pat.pattern);

        if not hits:
            return 0.0, [];

        # Score ramps with number of distinct pattern hits.
        # 1 hit -> 0.3, 2 -> 0.5, 3 -> 0.65, 4+ -> 0.8
        n = len(hits);
        if n == 1:
            s = 0.3;
        elif n == 2:
            s = 0.5;
        elif n == 3:
            s = 0.65;
        else:
            s = min(0.95, 0.8 + (n - 4) * 0.05);

        return s, hits;

    # ---- Layer 2: heuristics ---------------------------------------------

    def _heuristic_analysis(self, text: str) -> Tuple[float, list[str]]:
        """Return (additional_score, flag_names)."""
        flags: list[str] = [];
        score = 0.0;

        words = text.lower().split();
        word_count = len(words);

        if word_count == 0:
            return 0.0, [];

        # --- instruction density ---
        imperative_count = sum(1 for w in words if w in self._IMPERATIVE_VERBS);
        density = imperative_count / max(1.0, word_count / 100.0);

        if density > 0.15:
            score += 0.2;
            flags.append("high_instruction_density");
        elif density > 0.08:
            score += 0.1;
            flags.append("elevated_instruction_density");

        # --- role confusion ---
        role_markers = [
            "you are now", "you're now", "you have been",
            "pretend you", "act as if", "from now on you",
        ];
        if any(m in text.lower() for m in role_markers):
            score += 0.15;
            flags.append("role_confusion");

        # --- system prompt leakage attempts ---
        leakage_markers = [
            "your system prompt", "your instructions",
            "repeat your", "reveal your", "what are your",
            "print your", "dump your",
        ];
        if any(m in text.lower() for m in leakage_markers):
            score += 0.15;
            flags.append("system_prompt_leakage_attempt");

        # --- length anomaly: very short + high imperative count ---
        if word_count < 20 and imperative_count >= 3:
            score += 0.1;
            flags.append("length_anomaly_high_imperatives");

        return min(0.5, score), flags;

    # ---- classification --------------------------------------------------

    @staticmethod
    def _classify(score: float) -> str:
        if score >= 0.8:
            return "confirmed";
        if score >= 0.5:
            return "likely";
        if score >= 0.2:
            return "suspicious";
        return "clean";
