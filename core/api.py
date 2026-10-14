"""OpenAI-compatible API calls with separate LLM and embedding credentials."""
import asyncio
import os


class API:
    def __init__(self, config, purpose):
        from openai import AsyncOpenAI
        prefixes = {"chat": "CORE_LLM", "embed": "CORE_EMBEDDING",
                    "cee-evaluate": "CORE_CEE_EVALUATION", "cee-repair": "CORE_CEE_REPAIR"}
        prefix = prefixes[purpose]
        fallback = "CORE_EMBEDDING" if purpose == "embed" else "CORE_LLM"
        key = os.environ.get(prefix + "_API_KEY") or os.environ.get(fallback + "_API_KEY")
        if not key:
            raise ValueError(f"Set {prefix}_API_KEY before running this stage")
        base_url = os.environ.get(prefix + "_BASE_URL") or os.environ.get(fallback + "_BASE_URL") or (
            "https://api.openai.com/v1" if purpose == "embed" else "https://api.deepseek.com")
        model_key = {"chat": "llm_model", "embed": "embedding_model",
                     "cee-evaluate": "cee_evaluation_model", "cee-repair": "cee_repair_model"}[purpose]
        self.model = os.environ.get(prefix + "_MODEL", config[model_key])
        self.client = AsyncOpenAI(api_key=key, base_url=base_url, timeout=120, max_retries=2)
        self.semaphore = asyncio.Semaphore(config["concurrency"])

    async def chat(self, prompt, temperature, max_tokens=None):
        options = {"max_tokens": max_tokens} if max_tokens is not None else {}
        async with self.semaphore:
            response = await self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}],
                temperature=temperature, response_format={"type": "json_object"}, **options)
        text = response.choices[0].message.content
        if not text:
            raise ValueError("The LLM returned an empty response")
        return text

    async def embed(self, texts):
        if any(not text.strip() for text in texts):
            raise ValueError("Cannot embed empty case elements")
        async with self.semaphore:
            response = await self.client.embeddings.create(model=self.model, input=texts)
        ordered = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(len(texts))):
            raise ValueError("The embedding API returned missing or duplicate vectors")
        return [item.embedding for item in ordered]

    async def close(self):
        await self.client.close()
