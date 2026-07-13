# GPU0 G3 User Approval

- approval_time_utc: 2026-07-12T00:36:00Z
- workload: GPU0 DiCode DeepSeek-V4-Pro substitution baseline
- branch: exp/dicode-dspro
- approved_commit: f838946f2cb508bcdbf529b6394e7c1b753d3d61
- exact_model_id: deepseek-v4-pro
- requested_returned_identity_required: true
- physical_gpu: 0
- seed: 0
- max_environment_steps: 500000
- api_budget_mode: USER_AUTHORIZED_UNLIMITED
- api_budget_currency: N/A
- api_budget_value: UNLIMITED
- user_instruction: No monetary cap; start directly.
- mandatory_in_run_gate_steps: 50000
- output_collision_policy: REFUSE
- repair_limit: one smallest engineering repair; no scientific-result repair
- approved_phase: G3 pilot only
- excluded: G4, multi-seed, concurrent GPU0 runs

This approval is valid only for the bound branch, commit, exact model, GPU, seed, and step cap. Model mismatch, wrong GPU, CPU fallback, NaN/Inf, failed generation/compilation, secret exposure, output collision, missing manifest/accounting, or failed 50,000-step gate requires immediate stop.
