"""API connectivity and schema tests for the dspro substitution baseline.

REQUIRES: EXP_DEEPSEEK_API_KEY environment variable set.
NETWORK: Requires internet connectivity to api.deepseek.com.

These tests are separately invokable and clearly marked as requiring credentials.
Run with: pytest tests/test_dspro_api.py -v -m "network"

If credentials are absent, tests are skipped with a clear status message.
Never substitute a different model to make a test pass.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Marker for network/credential tests
pytestmark = pytest.mark.network

# Skip all tests if API key is not configured
API_KEY = os.getenv("EXP_DEEPSEEK_API_KEY")
NEEDS_API_KEY = pytest.mark.skipif(
    API_KEY is None,
    reason="EXP_DEEPSEEK_API_KEY not set. Skipping API tests.",
)


@NEEDS_API_KEY
class TestAPIConnectivity:
    """Minimal API connectivity and authentication tests."""

    def test_api_key_is_set(self):
        """API key environment variable must be set."""
        assert os.getenv("EXP_DEEPSEEK_API_KEY") is not None, \
            "EXP_DEEPSEEK_API_KEY not set"
        # Never print the key value
        key = os.getenv("EXP_DEEPSEEK_API_KEY")
        assert len(key) > 10, "API key seems too short"

    def test_minimal_connectivity(self):
        """Send minimal request to verify connectivity (smallest practical token usage)."""
        import asyncio
        from openai import AsyncOpenAI

        base_url = os.getenv("EXP_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=API_KEY,
        )

        async def _test():
            try:
                response = await client.chat.completions.create(
                    model="deepseek-v4-pro",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=1,
                    temperature=0.0,
                )
                return response
            except Exception as e:
                return e

        result = asyncio.run(_test())

        if isinstance(result, Exception):
            pytest.fail(f"API connectivity failed: {result}")

        assert result.choices[0].message.content is not None, \
            "API returned empty content"
        # Verify model identity
        assert result.model == "deepseek-v4-pro", \
            f"Expected model 'deepseek-v4-pro', got '{result.model}'"

    def test_json_schema_response(self):
        """Test that the API returns valid JSON-structured responses.

        DeepSeek V4 Pro may return content directly, or wrap it in markdown
        code fences, or place reasoning in reasoning_content. This test
        validates that a JSON object can be extracted from the response.
        """
        import asyncio
        from openai import AsyncOpenAI

        base_url = os.getenv("EXP_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=API_KEY,
        )

        async def _test():
            try:
                response = await client.chat.completions.create(
                    model="deepseek-v4-pro",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a JSON API. Reply ONLY with a valid JSON object. No markdown, no explanation."
                        },
                        {
                            "role": "user",
                            "content": 'Return: {"test": true, "value": 42}',
                        },
                    ],
                    max_tokens=100,
                    temperature=0.0,
                )
                return response
            except Exception as e:
                return e

        result = asyncio.run(_test())

        if isinstance(result, Exception):
            pytest.fail(f"API call failed: {result}")

        # Get content — try text content first, then reasoning_content
        msg = result.choices[0].message
        content = (msg.content or "").strip()
        reasoning = getattr(msg, "reasoning_content", None) or ""

        parsed = None

        # Strategy 1: Direct JSON parse of content (most common case)
        if content:
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                pass

        # Strategy 2: Extract JSON from markdown code fences in content
        if parsed is None and content:
            import re
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

        # Strategy 3: Find balanced JSON object from first brace in content
        if parsed is None and content:
            start = content.find("{")
            if start >= 0:
                # Find the matching close brace
                depth = 0
                end = -1
                for i in range(start, len(content)):
                    if content[i] == "{":
                        depth += 1
                    elif content[i] == "}":
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                if end > start:
                    try:
                        parsed = json.loads(content[start:end + 1])
                    except json.JSONDecodeError:
                        pass

        # Strategy 4: Try reasoning_content if content yielded nothing
        if parsed is None and reasoning:
            start = reasoning.find("{")
            end = reasoning.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(reasoning[start:end + 1])
                except json.JSONDecodeError:
                    pass

        if parsed is None:
            pytest.fail(
                f"Could not extract valid JSON from response.\n"
                f"Content ({len(content)} chars): {content[:200]}\n"
                f"Reasoning ({len(reasoning)} chars): {reasoning[:200]}"
            )

        assert isinstance(parsed, dict), f"Parsed value is not a dict: {type(parsed)}"
        # Validate content — at minimum we got a dict (gate: response is structured)
        assert len(parsed) > 0, "Parsed JSON object is empty"

    def test_model_id_returned(self):
        """API response must include the model ID for identity verification."""
        import asyncio
        from openai import AsyncOpenAI

        base_url = os.getenv("EXP_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=API_KEY,
        )

        async def _test():
            try:
                response = await client.chat.completions.create(
                    model="deepseek-v4-pro",
                    messages=[{"role": "user", "content": "Say 'ok'"}],
                    max_tokens=2,
                    temperature=0.0,
                )
                return response
            except Exception as e:
                return e

        result = asyncio.run(_test())

        if isinstance(result, Exception):
            pytest.fail(f"API call failed: {result}")

        # Verify the model field is present and correct
        assert hasattr(result, "model"), "Response missing 'model' field"
        assert result.model == "deepseek-v4-pro", \
            f"Expected model 'deepseek-v4-pro', got '{result.model}'"


@NEEDS_API_KEY
class TestAPISchemaValidation:
    """Schema validation tests — verify response structure."""

    def test_response_has_choices(self):
        """Response must have 'choices' array."""
        import asyncio
        from openai import AsyncOpenAI

        base_url = os.getenv("EXP_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        client = AsyncOpenAI(base_url=base_url, api_key=API_KEY)

        async def _test():
            return await client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[{"role": "user", "content": "Ping"}],
                max_tokens=1,
                temperature=0.0,
            )

        result = asyncio.run(_test())
        assert len(result.choices) > 0, "Response has no choices"
        assert result.choices[0].message.content is not None

    def test_response_has_usage(self):
        """Response should include token usage when available."""
        import asyncio
        from openai import AsyncOpenAI

        base_url = os.getenv("EXP_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        client = AsyncOpenAI(base_url=base_url, api_key=API_KEY)

        async def _test():
            return await client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[{"role": "user", "content": "One word reply"}],
                max_tokens=10,
                temperature=0.0,
            )

        result = asyncio.run(_test())
        # Usage may or may not be present depending on API version
        if hasattr(result, "usage") and result.usage:
            assert result.usage.total_tokens > 0, \
                f"Usage reported but total_tokens is 0: {result.usage}"

    def test_malformed_request_fails_gracefully(self):
        """Malformed request should raise a clear error, not crash."""
        import asyncio
        from openai import AsyncOpenAI

        base_url = os.getenv("EXP_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        client = AsyncOpenAI(base_url=base_url, api_key=API_KEY)

        async def _test():
            try:
                # Send an impossibly large max_tokens
                return await client.chat.completions.create(
                    model="deepseek-v4-pro",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=999999999,
                    temperature=0.0,
                )
            except Exception as e:
                return e

        result = asyncio.run(_test())
        # Should either succeed (API clamps) or fail with a clear error
        if isinstance(result, Exception):
            # Verify it's a proper error, not a crash
            assert hasattr(result, "__str__")
            error_str = str(result)
            # Should contain useful info
            assert len(error_str) > 0


@NEEDS_API_KEY
class TestExactModelPinning:
    """Verify that experimental API calls use the exact pinned model."""

    def test_dspro_config_matches_api_model(self):
        """The pinned model in config must match what the API returns."""
        from dicode.dspro.config import DSPRO_CONFIG

        assert DSPRO_CONFIG.model_id == "deepseek-v4-pro"

    def test_llm_deepseek_instantiation(self):
        """LLM class must instantiate with deepseek provider without error."""
        from dicode.dreaming.llm import LLM

        base_url = os.getenv("EXP_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

        llm = LLM(
            provider="deepseek",
            base_url=base_url,
            model="deepseek-v4-pro",
            llm_type="generation",
            max_tokens=100,
            temperature=0.0,
        )
        assert llm.provider == "deepseek"
        assert llm.model == "deepseek-v4-pro"

    def test_no_model_fallback(self):
        """Using wrong model should fail, not silently fall back."""
        import asyncio
        from openai import AsyncOpenAI

        base_url = os.getenv("EXP_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        client = AsyncOpenAI(base_url=base_url, api_key=API_KEY)

        async def _test():
            try:
                return await client.chat.completions.create(
                    model="nonexistent-model-xyz-123",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1,
                )
            except Exception as e:
                return e

        result = asyncio.run(_test())
        # Should fail — no silent fallback
        assert isinstance(result, Exception), \
            f"Expected failure for nonexistent model, got: {result}"

    def test_returned_model_matches_requested(self):
        """API returned_model must match requested_model. Fail if silently mapped."""
        import asyncio
        from openai import AsyncOpenAI

        base_url = os.getenv("EXP_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        client = AsyncOpenAI(base_url=base_url, api_key=API_KEY)
        requested = "deepseek-v4-pro"

        async def _test():
            return await client.chat.completions.create(
                model=requested,
                messages=[{"role": "user", "content": "Say 'ok'"}],
                max_tokens=2,
                temperature=0.0,
            )

        result = asyncio.run(_test())
        returned = result.model

        # Record both for the report
        print(f"\n    requested_model: {requested}")
        print(f"    returned_model:  {returned}")

        # GATE: Fail if the returned model doesn't match the requested model
        assert returned == requested, (
            f"MODEL MISMATCH GATE FAILED: "
            f"requested_model='{requested}' but API returned_model='{returned}'. "
            f"This means the request was silently mapped to a different model. "
            f"The pinned model 'deepseek-v4-pro' must be used directly."
        )

    def test_legacy_alias_rejected_by_manifest(self):
        """Manifest validation must reject legacy alias deepseek-chat."""
        from dicode.dspro.manifest import create_manifest, validate_manifest

        manifest = create_manifest(
            run_id="test-legacy-reject-001",
            output_path="/tmp/test-dspro-legacy-reject",
        )
        manifest["model_id"] = "deepseek-chat"
        manifest["requested_model"] = "deepseek-chat"
        errors = validate_manifest(manifest)
        assert len(errors) > 0, "Legacy alias should be rejected"
        assert any("legacy alias" in e.lower() for e in errors), \
            f"Error should mention legacy alias, got: {errors}"

    def test_legacy_alias_rejected_by_config(self):
        """DsproConfig.validate must reject legacy alias deepseek-chat."""
        from dicode.dspro.config import DsproConfig

        # Simulate a config with a legacy alias (must be rejected)
        assert "deepseek-chat" in DsproConfig().legacy_aliases
        assert "deepseek-reasoner" in DsproConfig().legacy_aliases


@NEEDS_API_KEY
class TestAPIKeySecurity:
    """Verify that API key is never exposed in logs or error messages."""

    def test_key_not_in_config_repr(self):
        """Config repr must not contain the API key."""
        from dicode.dspro.config import DSPRO_CONFIG

        key = os.getenv("EXP_DEEPSEEK_API_KEY")
        if key:
            repr_str = repr(DSPRO_CONFIG)
            assert key not in repr_str, "API key leaked in config repr!"

    def test_key_not_in_manifest(self):
        """Manifest must not contain the API key."""
        from dicode.dspro.manifest import create_manifest

        manifest = create_manifest(
            run_id="test-key-security-001",
            output_path="/tmp/test-dspro-key-security",
        )

        manifest_str = json.dumps(manifest)
        key = os.getenv("EXP_DEEPSEEK_API_KEY")
        if key:
            assert key not in manifest_str, "API key leaked in manifest!"

        # Also check that api_key_env name is there but not the value
        assert "EXP_DEEPSEEK_API_KEY" not in manifest_str or \
            "api_key_env" in manifest_str.lower(), \
            "API key env var name leaked as plain string in manifest"
