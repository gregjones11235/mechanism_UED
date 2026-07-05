import asyncio
import os
from typing import Any, Literal

from openai import AsyncOpenAI

class LLM:
	def __init__(
		self,
		provider: str,
		base_url: str,
		model: str,
		llm_type: Literal["generation", "embedding"],
		max_tokens: int = None,
		temperature: float = None,
		top_p: float = None,
		think: bool = False,
		embedding_size: int = 1024,
	):
		self.provider = provider
		self.base_url = base_url
		self.model = model
		self.llm_type = llm_type
		self.embedding_size = embedding_size

		# parameters for generation
		if self.llm_type == "generation":
			self.max_tokens = max_tokens
			self.temperature = temperature
			self.top_p = top_p
			self.think = think
		elif self.llm_type == "embedding":
			self.max_tokens = None
			self.temperature = None
			self.top_p = None
			self.think = False

		self.client = self._create_client()

	def _create_client(self):
		if self.provider == "local":
			return AsyncOpenAI(base_url=self.base_url, api_key="token-")
		elif self.provider == "gemini":
			api_key = os.getenv("GEMINI_API_KEY")
			return AsyncOpenAI(base_url=self.base_url, api_key=api_key)
		elif self.provider == "openai":
			api_key = os.getenv("OPENAI_API_KEY")
			return AsyncOpenAI(api_key=api_key)
		elif self.provider == "openrouter":
			api_key = os.getenv("OPENROUTER_API_KEY")
			# Default OpenRouter Base URL if not provided
			base_url = self.base_url or "https://openrouter.ai/api/v1"
			return AsyncOpenAI(
				base_url=base_url,
				api_key=api_key,
			)
		elif self.provider == "together":
			api_key = os.getenv("TOGETHER_API_KEY")
			# Together AI is OpenAI-compatible
			base_url = self.base_url or "https://api.together.xyz/v1"
			return AsyncOpenAI(
				base_url=base_url,
				api_key=api_key,
			)
		elif self.provider == "deepinfra":
			api_key = os.getenv("DEEPINFRA_API_KEY")
			# DeepInfra exposes an OpenAI-compatible endpoint
			base_url = self.base_url or "https://api.deepinfra.com/v1/openai"
			return AsyncOpenAI(
				base_url=base_url,
				api_key=api_key,
			)
		elif self.provider == "deepseek":
			# v6 cost-saving (v6模型省钱策略.md §1): DeepSeek OFFICIAL API. out -66% vs DeepInfra,
			# fastest TTFB (api.deepseek.com resolves to a US CDN edge), and prompt caching. Its own
			# key (DEEPSEEK_API_KEY) so we never silently reuse the DeepInfra key (§5 clean split).
			# OpenAI-compatible: same chat path as local/deepinfra below (query() whitelist).
			api_key = os.getenv("DEEPSEEK_API_KEY")
			base_url = self.base_url or "https://api.deepseek.com/v1"
			return AsyncOpenAI(
				base_url=base_url,
				api_key=api_key,
			)
		elif self.provider == "dashscope":
			# v6 cost-saving: Alibaba 百炼, 美国(弗吉尼亚) region 'dashscope-us' (user's key region,
			# verified 2026-07-04). BILLING IS 人民币 (跟账号主体走, not the region) — same RMB pricing
			# as 中国站. LATENCY WIN: Oscar(US) -> dashscope-us RTT 13.7ms / TTFB 0.15s — on par with
			# DeepSeek官方, FASTER than DeepInfra (191ms); no cross-border penalty (US-local node).
			# ⚠️ Keys are REGION-BOUND: this key only works on dashscope-us; 北京/新加坡 endpoints -> 401.
			api_key = os.getenv("DASHSCOPE_API_KEY")
			base_url = self.base_url or "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
			return AsyncOpenAI(
				base_url=base_url,
				api_key=api_key,
			)
		else:
			raise ValueError(f"Provider {self.provider} not supported")

	def _thinking_off_extra_body(self) -> dict[str, Any]:
		"""Vendor+PROVIDER-specific OFFICIAL way to turn reasoning OFF.

		v6 CRITICAL FIX: the switch depends on the PROVIDER, not just the model name. The SAME
		DeepSeek weights are served with DIFFERENT thinking-control APIs on DeepInfra (vLLM) vs the
		DeepSeek OFFICIAL endpoint. Dispatching by model name alone (as v5 did) sends the wrong
		switch after the v6 provider swap and either 400s or silently leaves thinking ON.

		Empirically verified from Oscar (2026-07-04, real key):
		  - DeepSeek OFFICIAL (api.deepseek.com): v4-pro/v4-flash default thinking ON. The ONLY switch
		    that turns it OFF is {"thinking": {"type": "disabled"}}. reasoning_effort:"none" -> 400
		    ("unknown variant none"); enable_thinking:false -> ignored (stays ON); thinking:false -> 400.
		  - DeepInfra (vLLM): DeepSeek off = reasoning_effort:"none"; Qwen/GLM off = enable_thinking:false
		    (reasoning_content 4594->0 Qwen3.5, 551->0 GLM-5.2). Unchanged from v5 so the GLM modeler,
		    which STAYS on DeepInfra in v6, behaves exactly as before.
		  - dashscope (Alibaba official, Qwen): [待核 pending real key] Qwen docs use enable_thinking:false.
		"""
		m = self.model.lower()
		if self.provider == "deepseek":
			# DeepSeek OFFICIAL endpoint — verified: only this disables thinking.
			return {"thinking": {"type": "disabled"}}
		if self.provider == "dashscope":
			# Alibaba official Qwen — enable_thinking switch (verify once dashscope key lands).
			return {"chat_template_kwargs": {"enable_thinking": False}}
		# DeepInfra (and other OpenAI-compatible vLLM hosts): unchanged v5 behaviour.
		if "deepseek" in m:
			return {"reasoning_effort": "none"}
		if "qwen" in m or "glm" in m or "zai" in m:
			return {"chat_template_kwargs": {"enable_thinking": False}}
		return {"reasoning_effort": "none"}

	def _thinking_on_extra_body(self) -> dict[str, Any]:
		"""Vendor+PROVIDER-specific OFFICIAL way to turn reasoning ON (mirror of the OFF variant).

		Verified from Oscar (2026-07-04):
		  - DeepSeek OFFICIAL: reasoning_effort:"high" yields reasoning_content (also default-ON).
		  - DeepInfra: DeepSeek reasoning_effort:"high"; Qwen/GLM enable_thinking:true (v5 behaviour,
		    kept for the GLM modeler which stays on DeepInfra).
		  - dashscope: [待核] enable_thinking:true.
		"""
		m = self.model.lower()
		if self.provider == "deepseek":
			return {"reasoning_effort": "high"}
		if self.provider == "dashscope":
			return {"chat_template_kwargs": {"enable_thinking": True}}
		if "deepseek" in m:
			return {"reasoning_effort": "high"}
		if "qwen" in m or "glm" in m or "zai" in m:
			return {"chat_template_kwargs": {"enable_thinking": True}}
		return {}

	async def _query_local_gen(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
		# Prompt text is identical whether or not thinking is on — thinking is toggled via the
		# per-model official extra_body switch below, NOT by polluting the user prompt.
		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		]
		extra_body = self._thinking_on_extra_body() if self.think else self._thinking_off_extra_body()

		try:
			chat_completion = await self.client.chat.completions.create(
				model=self.model,
				messages=messages,
				max_tokens=self.max_tokens,
				temperature=self.temperature,
				top_p=self.top_p,
				extra_body=extra_body,
			)

			return {
				"system_prompt": system_prompt,
				"user_prompt": user_prompt,
				"content": chat_completion.choices[0].message.content,
				"reasoning_content": getattr(chat_completion.choices[0].message, "reasoning_content", None),
				"error": None,
			}
		except Exception as e:
			return {
				"system_prompt": system_prompt,
				"user_prompt": user_prompt,
				"content": None,
				"reasoning_content": None,
				"error": e,
			}

	async def _query_with_retries(self, api_call_coroutine, max_retries=3, initial_delay=2):
		"""A wrapper to add exponential backoff retries to an API call."""
		for attempt in range(max_retries):
			try:
				# Await the actual API call coroutine
				result = await api_call_coroutine

				# Check for a valid content response
				if result.get("content") is not None and result.get("content").strip():
					return result  # Success

				# Handle cases where the API returns a valid response but empty content
				error_message = (
					f"LLM returned empty content. Attempt {attempt + 1} of {max_retries}."
				)
				print(f"Warning: {error_message}")
				result["error"] = ValueError(error_message)

			except Exception as e:
				# Handle network errors, API errors, etc.
				print(
					f"Warning: LLM API call failed with error: {e}. Attempt {attempt + 1} of {max_retries}."
				)
				result = {"content": None, "error": e}

			# If we're not on the last attempt, wait before retrying
			if attempt < max_retries - 1:
				await asyncio.sleep(initial_delay * (2**attempt))  # Exponential backoff
			else:
				print(f"Error: LLM call failed after {max_retries} retries.")
				return result  # Return the last failed result

	async def _query_local_embed(
		self, texts_to_embed: str | list[str] | tuple[str, ...], instruction: str = None
	) -> dict[str, Any]:
		if isinstance(texts_to_embed, str):
			input_list = [texts_to_embed]
			return_single_result = True
		elif isinstance(texts_to_embed, (list, tuple)):
			input_list = list(texts_to_embed)
			return_single_result = False
		else:
			raise ValueError(f"Invalid input type: {type(texts_to_embed)}")

		# --- Start of new code for instruction ---
		if instruction:
			formatted_input_list = [
				f"Instruct: {instruction}\nQuery: {text}" for text in input_list
			]
		else:
			formatted_input_list = input_list
		# --- End of new code for instruction ---

		if not formatted_input_list:
			return []
		try:
			response = await self.client.embeddings.create(
				model=self.model,
				input=formatted_input_list,  # Updated to use the formatted list
			)
			results = []
			for i, result in enumerate(response.data):
				sliced_embedding = result.embedding[: self.embedding_size]
				results.append(
					{
						"input_text": input_list[i],  # Use the original text for clarity
						"embedding": sliced_embedding,
						"embedding_dim": len(sliced_embedding),
						"error": None,
					}
				)
			return results
		except Exception as e:
			return [
				{
					"input_text": text,
					"embedding": None,
					"embedding_dim": 0,
					"error": str(e),
				}
				for text in input_list
			]

	async def _query_gemini(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		]

		try:
			chat_completion = await self.client.chat.completions.create(
				model=self.model,
				messages=messages,
				reasoning_effort="high",
			)

			return {
				"system_prompt": system_prompt,
				"user_prompt": user_prompt,
				"content": chat_completion.choices[0].message.content,
				"error": None,
			}
		except Exception as e:
			return {
				"system_prompt": system_prompt,
				"user_prompt": user_prompt,
				"content": None,
				"error": e,
			}

	async def _query_batch_local_gen(
		self, system_prompt: str, user_prompts: list[str]
	) -> list[dict[str, Any]]:
		tasks = [self._query_local_gen(system_prompt, prompt) for prompt in user_prompts]
		results = await asyncio.gather(*tasks)
		return results

	async def _query_batch_gemini(
		self, system_prompt: str, user_prompts: list[str]
	) -> list[dict[str, Any]]:
		tasks = [self._query_gemini(system_prompt, prompt) for prompt in user_prompts]
		results = await asyncio.gather(*tasks)
		return results

	def query(self, system_prompt: str, user_prompts: list[str]) -> list[dict[str, Any]]:
		if self.provider in ("local", "openai", "openrouter", "together", "deepinfra", "deepseek", "dashscope"):
			# All OpenAI-compatible chat-completion endpoints share the same batch path.
			return asyncio.run(self._query_batch_local_gen(system_prompt, user_prompts))
		elif self.provider == "gemini":
			return asyncio.run(self._query_batch_gemini(system_prompt, user_prompts))
		else:
			raise ValueError(f"Provider {self.provider} not supported")

	def get_embedding(
		self, text_to_embed: str | list[str] | tuple[str, ...], instruction: str = None
	) -> dict[str, Any]:
		if self.provider in ("local", "openai", "openrouter", "together", "deepinfra", "deepseek", "dashscope"):
			return asyncio.run(self._query_local_embed(text_to_embed, instruction))
		elif self.provider == "gemini":
			raise NotImplementedError("Gemini embeddings not yet implemented")
		else:
			raise ValueError(f"Provider {self.provider} not supported")
