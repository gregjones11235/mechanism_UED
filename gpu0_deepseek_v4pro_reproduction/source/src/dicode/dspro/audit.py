"""LLM call-site audit for DiCode DeepSeek-V4-Pro substitution baseline.

Traces every real experimental LLM invocation through wrappers, factories,
clients, and configuration resolution. Produces a complete inventory.

This is a model-substitution baseline.
It is not an exact reproduction under the original model conditions.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMCallSite:
    """Records one LLM call site in the DiCode codebase."""

    # Identity
    call_site_id: str
    source_file: str
    function_or_class: str
    logical_role: str  # Preserved role label

    # Provider and model
    provider_source: str  # Where the provider is configured
    model_source: str  # Where the model is configured
    current_provider: str
    current_model: str

    # Substitution
    substituted_provider: str = "deepseek"
    substituted_model: str = "deepseek-v4-pro"

    # Prompt
    prompt_source: str = ""  # Module path or description
    prompt_version: str = "migrated-v1"

    # Behavior
    retry_behavior: str = "none"
    max_retries: int = 0
    cache_behavior: str = "none"

    # Lifecycle
    lifecycle: str = "unknown"  # before_training, during_training, after_training

    # Notes
    in_ppo_update_loop: bool = False
    notes: str = ""

    # API accounting
    estimated_calls_per_session: int = 0


def audit_call_sites() -> list[LLMCallSite]:
    """Return the complete inventory of experimental DiCode LLM call sites.

    Each call site is traced through its full resolution path:
    config → GenManager → LLM class → API call.

    Returns:
        List of LLMCallSite records.
    """
    sites: list[LLMCallSite] = []

    # ==========================================================================
    # Call Site 1: Task Generator (task description generation)
    # ==========================================================================
    sites.append(LLMCallSite(
        call_site_id="CS-001",
        source_file="src/dicode/dreaming/gen_manager.py",
        function_or_class="GenManager.__init__ → TaskGenerator.__init__ → evolve_mastered/evolve_ablation → _query_and_parse_responses → LLM.query",
        logical_role="task_generator",
        provider_source="conf/gen_manager/dspro_substitution.yaml → task_generator.provider",
        model_source="conf/gen_manager/llm/deepseek.yaml → model",
        current_provider="deepseek",
        current_model="deepseek-v4-pro",
        prompt_source="dicode.dreaming.prompts.dicode.evolve / dicode.dreaming.prompts.cl_.evolve_mastered_r",
        prompt_version="migrated-v1",
        retry_behavior="exponential_backoff",
        max_retries=3,  # LLM class default + 10 parse retries
        cache_behavior="none",
        lifecycle="before_training",
        in_ppo_update_loop=False,
        estimated_calls_per_session=5,  # num_generations
        notes="Runs in background ThreadPoolExecutor worker. Generates task descriptions "
              "from mastered tasks. Parse retries (max 10) re-query on failure. "
              "Not inside PPO update loop.",
    ))

    # ==========================================================================
    # Call Site 2: Env Generator (environment code generation)
    # ==========================================================================
    sites.append(LLMCallSite(
        call_site_id="CS-002",
        source_file="src/dicode/dreaming/gen_manager.py",
        function_or_class="GenManager.__init__ → EnvGenerator.__init__ → generate/generate_code_only → LLM.query",
        logical_role="env_generator",
        provider_source="conf/gen_manager/dspro_substitution.yaml → env_generator.provider",
        model_source="conf/gen_manager/llm/deepseek.yaml → model",
        current_provider="deepseek",
        current_model="deepseek-v4-pro",
        prompt_source="dicode.dreaming.prompts.cl_.gen_env",
        prompt_version="migrated-v1",
        retry_behavior="exponential_backoff_and_reflection_loop",
        max_retries=3,  # LLM class + unlimited reflection loop until compile
        cache_behavior="none",
        lifecycle="before_training",
        in_ppo_update_loop=False,
        estimated_calls_per_session=10,  # Initial generation + reflection rounds
        notes="Runs in background ThreadPoolExecutor worker (generate_code_only) "
              "and main thread (generate with compilation). "
              "Reflection loop may cause many calls per task until compilation succeeds. "
              "Not inside PPO update loop.",
    ))

    # ==========================================================================
    # Call Site 3: Embedding Model (task similarity via embeddings)
    # ==========================================================================
    sites.append(LLMCallSite(
        call_site_id="CS-003",
        source_file="src/dicode/dreaming/gen_manager.py",
        function_or_class="TaskSelector._order_similar_tasks → embedding_model.get_embedding",
        logical_role="embedding_model",
        provider_source="conf/gen_manager/dspro_substitution.yaml → embedding_model.provider",
        model_source="conf/gen_manager/llm/local_embed.yaml → model",
        current_provider="local",
        current_model="Qwen/Qwen3-Embedding-0.6B",
        substituted_provider="local",  # DeepSeek has no embeddings API
        substituted_model="Qwen/Qwen3-Embedding-0.6B",
        prompt_source="N/A (embedding instruction string)",
        prompt_version="migrated-v1",
        retry_behavior="none",
        max_retries=0,
        cache_behavior="none",
        lifecycle="before_training",
        in_ppo_update_loop=False,
        estimated_calls_per_session=3,
        notes="NOT a generation LLM call. Embedding model kept as local_embed because "
              "DeepSeek does not provide an embeddings API. Used for task similarity "
              "ordering during curriculum selection. Not inside PPO update loop.",
    ))

    # ==========================================================================
    # Call Site 4: Session Embedding Generation
    # ==========================================================================
    sites.append(LLMCallSite(
        call_site_id="CS-004",
        source_file="src/dicode/training.py",
        function_or_class="_generate_embeddings_for_session → gen_manager.selector.embedding_model.get_embedding",
        logical_role="embedding_model",
        provider_source="conf/gen_manager/dspro_substitution.yaml → embedding_model.provider",
        model_source="conf/gen_manager/llm/local_embed.yaml → model",
        current_provider="local",
        current_model="Qwen/Qwen3-Embedding-0.6B",
        substituted_provider="local",
        substituted_model="Qwen/Qwen3-Embedding-0.6B",
        prompt_source="EMBEDDING_INSTRUCTION constant",
        prompt_version="migrated-v1",
        retry_behavior="none",
        max_retries=0,
        cache_behavior="none",
        lifecycle="before_training",
        in_ppo_update_loop=False,
        estimated_calls_per_session=1,
        notes="Generates embeddings for the session's task classes. "
              "Uses same embedding model as CS-003. Not inside PPO update loop.",
    ))

    # ==========================================================================
    # Call Site 5: Seed Training Embeddings
    # ==========================================================================
    sites.append(LLMCallSite(
        call_site_id="CS-005",
        source_file="src/dicode/setup.py",
        function_or_class="_generate_task_embeddings → gen_manager.selector.embedding_model.get_embedding",
        logical_role="embedding_model",
        provider_source="conf/gen_manager/dspro_substitution.yaml → embedding_model.provider",
        model_source="conf/gen_manager/llm/local_embed.yaml → model",
        current_provider="local",
        current_model="Qwen/Qwen3-Embedding-0.6B",
        substituted_provider="local",
        substituted_model="Qwen/Qwen3-Embedding-0.6B",
        prompt_source="EMBEDDING_INSTRUCTION constant",
        prompt_version="migrated-v1",
        retry_behavior="none",
        max_retries=0,
        cache_behavior="none",
        lifecycle="before_training",
        in_ppo_update_loop=False,
        estimated_calls_per_session=1,
        notes="One-time embedding generation during seed training phase. "
              "Not inside PPO update loop.",
    ))

    # ==========================================================================
    # Call Site 6: Role Judge — Tutor (curriculum judgment, batch/offline)
    # ==========================================================================
    sites.append(LLMCallSite(
        call_site_id="CS-006",
        source_file="src/dicode/mechanisms/llm_roles.py",
        function_or_class="call_role_judge → call_llm_api",
        logical_role="tutor",
        provider_source="conf/llm/providers.yaml → ROLE_PROVIDER_MAP['tutor']",
        model_source="conf/llm/providers.yaml → model_catalog",
        current_provider="qwen",  # Original role mapping
        current_model="qwen-turbo",  # Original default
        prompt_source="src/dicode/mechanisms/llm_roles.py → build_role_prompt('tutor', ...)",
        prompt_version="migrated-v1",
        retry_behavior="none_at_api_level",
        max_retries=0,
        cache_behavior="jsonl_disk_cache",
        lifecycle="after_training",
        in_ppo_update_loop=False,
        estimated_calls_per_session=0,
        notes="Called by offline aggregation/judgment scripts, not during training. "
              "Results cached to JSONL and read by select_tasks_with_aggregation. "
              "For dspro substitution: this role should be re-routed to deepseek-v4-pro. "
              "Not inside PPO update loop.",
    ))

    # ==========================================================================
    # Call Site 7: Role Judge — Critic (curriculum judgment, batch/offline)
    # ==========================================================================
    sites.append(LLMCallSite(
        call_site_id="CS-007",
        source_file="src/dicode/mechanisms/llm_roles.py",
        function_or_class="call_role_judge → call_llm_api",
        logical_role="critic",
        provider_source="conf/llm/providers.yaml → ROLE_PROVIDER_MAP['critic']",
        model_source="conf/llm/providers.yaml → model_catalog",
        current_provider="deepseek",  # Already deepseek in original
        current_model="deepseek-v4-pro",  # Original default
        prompt_source="src/dicode/mechanisms/llm_roles.py → build_role_prompt('critic', ...)",
        prompt_version="migrated-v1",
        retry_behavior="none_at_api_level",
        max_retries=0,
        cache_behavior="jsonl_disk_cache",
        lifecycle="after_training",
        in_ppo_update_loop=False,
        estimated_calls_per_session=0,
        notes="Already mapped to deepseek in original configuration. "
              "For dspro baseline: same model, but pinned explicitly. "
              "Not inside PPO update loop.",
    ))

    # ==========================================================================
    # Call Site 8: Role Judge — Explorer (curriculum judgment, batch/offline)
    # ==========================================================================
    sites.append(LLMCallSite(
        call_site_id="CS-008",
        source_file="src/dicode/mechanisms/llm_roles.py",
        function_or_class="call_role_judge → call_llm_api",
        logical_role="explorer",
        provider_source="conf/llm/providers.yaml → ROLE_PROVIDER_MAP['explorer']",
        model_source="conf/llm/providers.yaml → model_catalog",
        current_provider="glm",  # Original role mapping
        current_model="glm-4-flash",  # Original default
        prompt_source="src/dicode/mechanisms/llm_roles.py → build_role_prompt('explorer', ...)",
        prompt_version="migrated-v1",
        retry_behavior="none_at_api_level",
        max_retries=0,
        cache_behavior="jsonl_disk_cache",
        lifecycle="after_training",
        in_ppo_update_loop=False,
        estimated_calls_per_session=0,
        notes="Called by offline aggregation/judgment scripts, not during training. "
              "For dspro substitution: this role should be re-routed to deepseek-v4-pro. "
              "Not inside PPO update loop.",
    ))

    return sites


def generate_audit_report(sites: list[LLMCallSite]) -> str:
    """Generate a markdown audit report from call site inventory.

    Args:
        sites: List of LLMCallSite records.

    Returns:
        Markdown formatted report string.
    """
    lines = [
        "# DiCode LLM Call-Site Audit — DeepSeek V4 Pro Substitution Baseline",
        "",
        f"**Total call sites identified:** {len(sites)}",
        "",
        "## Generation LLM Call Sites (routed through deepseek-v4-pro)",
        "",
    ]

    gen_sites = [s for s in sites if s.substituted_provider == "deepseek"]
    for s in gen_sites:
        lines.extend([
            f"### {s.call_site_id}: {s.logical_role}",
            "",
            f"- **Source file:** `{s.source_file}`",
            f"- **Function/class:** `{s.function_or_class}`",
            f"- **Logical role:** `{s.logical_role}`",
            f"- **Current provider:** `{s.current_provider}`",
            f"- **Current model:** `{s.current_model}`",
            f"- **Substituted provider:** `{s.substituted_provider}`",
            f"- **Substituted model:** `{s.substituted_model}`",
            f"- **Lifecycle:** `{s.lifecycle}`",
            f"- **In PPO update loop:** {s.in_ppo_update_loop}",
            f"- **Retry behavior:** {s.retry_behavior} (max {s.max_retries})",
            f"- **Cache behavior:** {s.cache_behavior}",
            f"- **Est. calls/session:** {s.estimated_calls_per_session}",
            f"- **Notes:** {s.notes}",
            "",
        ])

    emb_sites = [s for s in sites if s.substituted_provider != "deepseek"]
    if emb_sites:
        lines.append("## Non-Generation Call Sites (separate provider)")
        lines.append("")
        for s in emb_sites:
            lines.extend([
                f"### {s.call_site_id}: {s.logical_role}",
                "",
                f"- **Provider:** `{s.current_provider}` (Kept as-is — DeepSeek has no embeddings API)",
                f"- **Model:** `{s.current_model}`",
                f"- **Lifecycle:** `{s.lifecycle}`",
                f"- **In PPO update loop:** {s.in_ppo_update_loop}",
                f"- **Notes:** {s.notes}",
                "",
            ])

    lines.extend([
        "## Summary",
        "",
        f"- **Generation LLM call sites routed through deepseek-v4-pro:** {len(gen_sites)}",
        f"- **Non-generation call sites (separate provider):** {len(emb_sites)}",
        f"- **Call sites inside PPO update loop:** {sum(1 for s in sites if s.in_ppo_update_loop)}",
        "- **All LLM calls occur before or after PPO training, never inside the update loop.**",
        "",
        "## Key Findings",
        "",
        "1. No experimental LLM API call occurs inside the JAX PPO update loop.",
        "2. Task Generator (CS-001) and Env Generator (CS-002) run in background threads before training sessions.",
        "3. Embedding calls (CS-003, CS-004, CS-005) use local_embed because DeepSeek has no embeddings API.",
        "4. Role judgment calls (CS-006, CS-007, CS-008) are offline/batch operations, cached to JSONL.",
        "5. For full dspro substitution, role judgment scripts must be reconfigured to use deepseek-v4-pro for all three roles.",
    ])

    return "\n".join(lines)
