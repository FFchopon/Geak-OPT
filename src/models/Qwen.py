from typing import List, Optional
import os
from tenacity import retry, stop_after_attempt, wait_random_exponential

import requests

from models.Base import BaseModel


class QwenModel(BaseModel):
    def __init__(
        self,
        model_id: str = "qwen3-coder-plus",
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ):
        if api_key is None:
            api_key = os.environ.get("DASHSCOPE_API_KEY")
        assert api_key is not None, "no api key is provided. Please set DASHSCOPE_API_KEY or pass api_key."

        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

        self._client = None
        try:
            from openai import OpenAI  # type: ignore

            self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        except Exception:
            self._client = None

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(
        self,
        messages: List,
        temperature=1.0,
        presence_penalty=0,
        frequency_penalty=0,
        max_tokens=8192,
        **kwargs,
    ) -> str:
        try:
            max_tokens_int = int(max_tokens)
        except Exception:
            max_tokens_int = 8192
        max_tokens = max(1, min(8192, max_tokens_int))

        if self._client is not None:
            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
            if not response or not hasattr(response, "choices") or len(response.choices) == 0:
                raise ValueError("No response choices returned from the API.")
            msg = response.choices[0].message
            return getattr(msg, "content", None) or ""

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "stream": False,
        }
        resp = requests.post(url, json=body, headers=headers, timeout=600)
        if resp.status_code != 200:
            raise ValueError(f"Qwen API error {resp.status_code}: {resp.text}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            raise ValueError(f"Unexpected Qwen response schema: {data}")
