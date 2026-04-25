"""
Multi-Model Roundtable — Integrated into AI Inference Gateway.

Orchestrates multi-round deliberations between different LLM models,
leveraging the gateway's circuit breakers, model routing, RAG, and caching.

Architecture:
  Round 1: Each model states independent position (parallel)
  Round 2+: Each model sees ALL positions → critique + refine (parallel)
  Synthesis: Moderator model reads everything → structured output

All LLM calls go through state.openai_client — so they get:
  - Automatic backend failover (local → NIM → ZAI → Pollinations)
  - Circuit breaker protection
  - Model-aware routing (which GPU, which provider)
  - Response sanitization (MLSEC pipeline)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class RoundtableModelConfig(BaseModel):
    """Configuration for a single model participant."""
    model: str = Field(..., description="Gateway model ID (from /v1/models)")
    name: Optional[str] = Field(None, description="Display name for this participant")
    persona: Optional[str] = Field(None, description="System prompt persona override")
    temperature: float = Field(0.7, description="Sampling temperature")
    max_tokens: int = Field(2048, description="Max response tokens")


class RoundtableRequest(BaseModel):
    """Request body for roundtable deliberation."""
    topic: str = Field(..., description="Topic/question for deliberation")
    models: Optional[List[RoundtableModelConfig]] = Field(
        None,
        description="Model participants. If omitted, uses default panel.",
    )
    rounds: int = Field(2, description="Number of deliberation rounds", ge=1, le=5)
    context: Optional[str] = Field(None, description="Additional context string")
    use_rag: bool = Field(
        False,
        description="Enrich deliberation with RAG (Knowledge Fabric) search on the topic",
    )
    rag_collection: str = Field("brain-wiki", description="Qdrant collection for RAG context")
    rag_max_results: int = Field(5, description="Max RAG results to inject")
    synthesis_model: Optional[str] = Field(
        None,
        description="Model ID for synthesis. Defaults to first model in panel.",
    )
    stream: bool = Field(
        True,
        description="Stream round-by-round progress via SSE",
    )


class RoundResponse(BaseModel):
    """Single model response within a round."""
    model_name: str
    model_id: str
    content: str
    round_num: int
    latency_ms: float = 0.0
    error: Optional[str] = None


class RoundtableResponse(BaseModel):
    """Complete roundtable response."""
    topic: str
    rounds: List[List[RoundResponse]] = Field(default_factory=list)
    synthesis: Optional[str] = None
    rag_context: Optional[str] = None
    models_used: List[str] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    cached: bool = False


# ---------------------------------------------------------------------------
# Default Panel (models that exist in the gateway router)
# ---------------------------------------------------------------------------

DEFAULT_PANEL = [
    RoundtableModelConfig(
        model="meta/llama-3.3-70b-instruct",
        name="Llama-3.3-70B",
        persona=(
            "You are LLAMA, a battle-tested automation engineer. "
            "Focus on what ACTUALLY works in production, not what looks good in a README. "
            "Be specific about real-world failures and gotchas."
        ),
    ),
    RoundtableModelConfig(
        model="deepseek-ai/deepseek-v4-flash",
        name="DeepSeek-V4",
        persona=(
            "You are DEEPSEEK, a deeply analytical researcher. "
            "You pick apart arguments logically, find hidden assumptions, "
            "and stress-test every claim with edge cases."
        ),
    ),
    RoundtableModelConfig(
        model="google/gemma-4-31b-it",
        name="Gemma-4-31B",
        persona=(
            "You are GEMMA, a pragmatic systems architect. "
            "Design for simplicity, maintainability, and real constraints. "
            "Hate over-engineering. Love elegant minimal solutions."
        ),
    ),
    RoundtableModelConfig(
        model="qwen/qwen3-coder-480b-a35b-instruct",
        name="Qwen3-Coder",
        persona=(
            "You are QWEN, a cutting-edge AI researcher who stays current on the latest "
            "tools and frameworks. You bring fresh perspectives and challenge conventional wisdom."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

@dataclass
class InternalRoundResult:
    """Internal result tracking during deliberation."""
    round_num: int
    model_name: str
    model_id: str
    response: str
    latency_ms: float = 0.0
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


def _build_round_messages(
    model_cfg: RoundtableModelConfig,
    topic: str,
    round_num: int,
    previous_rounds: List[List[InternalRoundResult]],
    context: str = "",
) -> List[Dict[str, str]]:
    """Build the message list for a model in a given round."""
    system = model_cfg.persona or "You are an expert AI participant in a multi-model deliberation."

    if context:
        system += f"\n\n## Additional Context\n{context}"

    messages = [{"role": "system", "content": system}]

    if round_num == 1:
        messages.append({
            "role": "user",
            "content": (
                f"## Topic for Deliberation\n\n{topic}\n\n"
                "Provide your independent analysis. Be opinionated. "
                "Defend your position with specific technical details. "
                "Critique approaches you disagree with."
            ),
        })
    else:
        round_contexts = []
        for prev_round in previous_rounds:
            for r in prev_round:
                round_contexts.append(f"**{r.model_name}** (Round {r.round_num}):\n{r.response}")

        combined = "\n\n---\n\n".join(round_contexts)

        messages.append({
            "role": "user",
            "content": (
                f"## Topic for Deliberation\n\n{topic}\n\n"
                f"## Previous Round{'s' if round_num > 2 else ''} Summary\n\n"
                f"{combined}\n\n"
                f"---\n\n"
                f"You are now in **Round {round_num}**. Review what the other models said. "
                f"Respond to their arguments. Strengthen your position or change it if "
                f"persuaded. Find points of agreement and disagreement. "
                f"Be specific — cite technical details from their responses."
            ),
        })

    return messages


async def _call_model(
    openai_client,
    model_id: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> tuple[str, float, Optional[str]]:
    """
    Call a model through the gateway's OpenAI client wrapper.

    Returns (content, latency_ms, error) tuple.
    The openai_client handles backend failover, circuit breakers, and routing.
    """
    start = time.time()
    try:
        # Use the gateway's chat_completion which handles routing + failover
        response = await openai_client.chat_completion(
            messages=messages,
            model=model_id,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = ""
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if choice.message and choice.message.content:
                content = choice.message.content
            elif hasattr(choice, 'message') and hasattr(choice.message, 'reasoning_content'):
                # Some models return content in reasoning_content when thinking is on
                rc = getattr(choice.message, 'reasoning_content', None)
                if rc:
                    content = rc

        latency_ms = (time.time() - start) * 1000
        return content, latency_ms, None

    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        logger.error(f"Roundtable model call failed for {model_id}: {e}")
        return "", latency_ms, str(e)


async def _run_round(
    openai_client,
    models: List[RoundtableModelConfig],
    topic: str,
    round_num: int,
    previous_rounds: List[List[InternalRoundResult]],
    context: str = "",
) -> List[InternalRoundResult]:
    """Run one round — all models respond in parallel."""
    tasks = []
    for m in models:
        messages = _build_round_messages(m, topic, round_num, previous_rounds, context)
        tasks.append(_call_model(
            openai_client,
            m.model,
            messages,
            temperature=m.temperature,
            max_tokens=m.max_tokens,
        ))

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for m, resp in zip(models, responses):
        if isinstance(resp, Exception):
            results.append(InternalRoundResult(
                round_num=round_num,
                model_name=m.name or m.model,
                model_id=m.model,
                response="",
                error=str(resp),
            ))
        else:
            content, latency_ms, error = resp
            results.append(InternalRoundResult(
                round_num=round_num,
                model_name=m.name or m.model,
                model_id=m.model,
                response=content,
                latency_ms=latency_ms,
                error=error,
            ))

    return results


async def _run_synthesis(
    openai_client,
    model_id: str,
    topic: str,
    all_rounds: List[List[InternalRoundResult]],
    context: str = "",
) -> str:
    """Run final synthesis — one model summarizes the deliberation."""
    transcript_parts = []
    for round_results in all_rounds:
        for r in round_results:
            if r.error:
                transcript_parts.append(
                    f"### {r.model_name} (Round {r.round_num}) — ERROR\n\n{r.error}"
                )
            else:
                transcript_parts.append(
                    f"### {r.model_name} (Round {r.round_num})\n\n{r.response}"
                )

    transcript = "\n\n---\n\n".join(transcript_parts)

    context_block = ""
    if context:
        context_block = f"\n\n## Additional Context\n\n{context}\n\n---\n\n"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a neutral moderator synthesizing a multi-model AI deliberation. "
                "Your job is to produce a clear, actionable summary that a human decision-maker "
                "can act on immediately."
            ),
        },
        {
            "role": "user",
            "content": (
                f"## Topic\n\n{topic}\n\n"
                f"{context_block}"
                f"## Full Deliberation Transcript\n\n{transcript}\n\n"
                f"---\n\n"
                f"## Your Task\n\n"
                f"Produce a structured synthesis with:\n"
                f"1. **Areas of Consensus** — where models agreed\n"
                f"2. **Key Disagreements** — where models differed and why\n"
                f"3. **Top Recommendation** — the clearest winning approach\n"
                f"4. **Critical Risks** — what could go wrong\n"
                f"5. **Action Items** — concrete next steps\n"
            ),
        },
    ]

    content, _, error = await _call_model(
        openai_client,
        model_id,
        messages,
        temperature=0.3,
        max_tokens=4096,
    )

    if error:
        return f"[Synthesis Error]: {error}"

    return content or "[Synthesis returned empty]"


async def run_roundtable(
    state,
    request: RoundtableRequest,
) -> RoundtableResponse:
    """
    Execute a full roundtable deliberation using the gateway's infrastructure.

    Args:
        state: GatewayState with openai_client, rag_search, router, etc.
        request: RoundtableRequest configuration

    Returns:
        RoundtableResponse with all rounds + synthesis
    """
    start_time = time.time()

    # Resolve model panel
    models = request.models if request.models else DEFAULT_PANEL

    # Resolve display names if not set
    for m in models:
        if not m.name:
            m.name = m.model.split("/")[-1]

    # Optional RAG context enrichment
    rag_context = None
    if request.use_rag and state.rag_search:
        try:
            rag_results = await state.rag_search.search(
                query=request.topic,
                max_results=request.rag_max_results,
                collection=request.rag_collection,
            )
            if rag_results and rag_results.get("results"):
                context_parts = []
                for r in rag_results["results"][:request.rag_max_results]:
                    content = r.get("content", r.get("text", ""))
                    source = r.get("metadata", {}).get("source", r.get("title", "unknown"))
                    score = r.get("score", r.get("dense_score", 0))
                    context_parts.append(f"[{source}] (score: {score:.3f}):\n{content}")
                rag_context = "\n\n---\n\n".join(context_parts)
                logger.info(f"Roundtable RAG enrichment: {len(context_parts)} results")
        except Exception as e:
            logger.warning(f"Roundtable RAG enrichment failed (non-fatal): {e}")

    # Combine context sources
    full_context = ""
    if request.context:
        full_context += request.context
    if rag_context:
        if full_context:
            full_context += "\n\n---\n\n## Knowledge Fabric Results\n\n" + rag_context
        else:
            full_context = "## Knowledge Fabric Results\n\n" + rag_context

    # Run deliberation rounds
    all_rounds: List[List[InternalRoundResult]] = []

    for round_num in range(1, request.rounds + 1):
        logger.info(f"Roundtable round {round_num}/{request.rounds}")

        results = await _run_round(
            state.openai_client,
            models,
            request.topic,
            round_num,
            all_rounds,
            full_context,
        )

        all_rounds.append(results)

        # Brief pause between rounds (rate-limit courtesy)
        if round_num < request.rounds:
            await asyncio.sleep(1)

    # Synthesis
    synth_model_id = request.synthesis_model or models[0].model
    logger.info(f"Roundtable synthesis using {synth_model_id}")

    synthesis = await _run_synthesis(
        state.openai_client,
        synth_model_id,
        request.topic,
        all_rounds,
        full_context,
    )

    total_ms = (time.time() - start_time) * 1000

    # Build response
    response_rounds = []
    for round_results in all_rounds:
        response_rounds.append([
            RoundResponse(
                model_name=r.model_name,
                model_id=r.model_id,
                content=r.response,
                round_num=r.round_num,
                latency_ms=r.latency_ms,
                error=r.error,
            )
            for r in round_results
        ])

    return RoundtableResponse(
        topic=request.topic,
        rounds=response_rounds,
        synthesis=synthesis,
        rag_context=rag_context,
        models_used=[m.model for m in models],
        total_latency_ms=total_ms,
    )


def format_roundtable_markdown(response: RoundtableResponse) -> str:
    """Format a RoundtableResponse as markdown for file output."""
    parts = [
        "# Multi-Model Roundtable Deliberation\n",
        f"## Topic\n\n{response.topic}\n",
        f"## Participants\n",
    ]

    seen = set()
    for round_list in response.rounds:
        for r in round_list:
            if r.model_name not in seen:
                seen.add(r.model_name)
                parts.append(f"- **{r.model_name}** (`{r.model_id}`)")

    parts.append("")

    if response.rag_context:
        parts.append("---\n## Knowledge Fabric Context\n")
        parts.append(response.rag_context)
        parts.append("")

    for round_list in response.rounds:
        if round_list:
            parts.append(f"---\n## Round {round_list[0].round_num}\n")
            for r in round_list:
                parts.append(f"### {r.model_name}\n")
                if r.error:
                    parts.append(f"*[Error: {r.error}]*\n")
                else:
                    parts.append(r.content)
                    parts.append(f"\n*(latency: {r.latency_ms:.0f}ms)*\n")
                parts.append("")

    parts.append("---\n## Synthesis\n")
    parts.append(response.synthesis or "[No synthesis]")

    parts.append(f"\n\n---\n*Total latency: {response.total_latency_ms:.0f}ms | Models: {', '.join(response.models_used)}*")

    return "\n".join(parts)
