"""
llm_clients.py
==============

统一的 LLM 客户端抽象层，支持多种商业与国产大模型：

    - OpenAI / GPT            (OpenAI SDK 原生 `/v1/responses` or `/v1/chat/completions`)
    - Anthropic / Claude      (anthropic SDK)
    - Google / Gemini         (google-generativeai SDK)
    - Alibaba Qwen (通义千问)  (dashscope compatible-mode / OpenAI SDK)
    - MiniMax (海螺)          (OpenAI 兼容 HTTP 接口)
    - DeepSeek (V3 / R1)      (DeepSeek 官方 OpenAI 兼容 endpoint)

设计要点：
    1. `BaseLLMClient` 暴露统一的 `generate(prompt, system_instruction, n) -> List[str]`
       接口——这是系统中唯一被各模块依赖的契约；
    2. 每个后端实现只负责把这个统一接口映射到自身 SDK 细节；
    3. `LLMClientFactory.create(provider, **kwargs)` 用字符串名字创建任意后端，
       方便通过配置/环境变量切换；
    4. 所有后端共享：温度控制、重试、超时、n 份独立响应采样、
       系统指令与用户 prompt 合并策略。

使用方式：
    >>> from llm_clients import LLMClientFactory
    >>> client = LLMClientFactory.create("openai", api_key="...", model="your-model")
    >>> outs = client.generate("Hello", "You are helpful.", n=2)
"""

from __future__ import annotations

import os
import time
import json
import logging
import signal
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# ==========================================
# 1. 统一配置与基类
# ==========================================

@dataclass
class LLMConfig:
    """所有后端客户端共用的配置入口。具体后端可在此基础上扩展参数。"""

    api_key: str
    model: str                      # 模型名称 / Endpoint ID
    base_url: Optional[str] = None  # 自定义网关
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    timeout: float = 120.0
    max_retries: int = 2
    extra: Dict[str, Any] = field(default_factory=dict)  # 厂商特定参数透传


class BaseLLMClient(ABC):
    """
    所有 LLM 后端必须实现的基类。
    为所有 Synthesizer / Evaluator / Memory / StrategyAdapter 提供统一入口。
    """

    provider_name: str = "base"

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def _single_call(self, prompt: str, system_instruction: str) -> str:
        """子类只需实现单次调用；`generate` 会负责多次采样 + 重试。"""

    def generate(self, prompt: str, system_instruction: str, n: int = 1) -> List[str]:
        """
        生成 n 份独立响应。对 n>1，通过多次独立调用来保证 diversity。
        包含指数退避重试。
        """
        outputs: List[str] = []
        for i in range(n):
            attempt = 0
            last_err: Optional[Exception] = None
            while attempt <= self.config.max_retries:
                try:
                    text = self._single_call_with_hard_timeout(prompt, system_instruction)
                    outputs.append(text.strip())
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    wait = 1.5 ** attempt
                    logger.warning(
                        "[%s] call failed (attempt %d/%d): %s. Retrying in %.1fs",
                        self.provider_name, attempt + 1, self.config.max_retries + 1, e, wait
                    )
                    time.sleep(wait)
                    attempt += 1
            else:
                logger.error("[%s] gave up after %d retries: %s",
                             self.provider_name, self.config.max_retries, last_err)
                outputs.append("")
        return outputs

    def _single_call_with_hard_timeout(self, prompt: str, system_instruction: str) -> str:
        """Apply a POSIX wall-clock timeout around SDK calls that may ignore their own timeout."""
        timeout = float(self.config.timeout or 0)
        if (
            timeout <= 0
            or threading.current_thread() is not threading.main_thread()
            or not hasattr(signal, "SIGALRM")
        ):
            return self._single_call(prompt, system_instruction)

        def _on_timeout(signum, frame):  # noqa: ARG001
            raise TimeoutError(f"LLM call exceeded {timeout:.1f}s")

        old_handler = signal.signal(signal.SIGALRM, _on_timeout)
        try:
            signal.setitimer(signal.ITIMER_REAL, timeout)
            return self._single_call(prompt, system_instruction)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old_handler)

    @staticmethod
    def _merge_prompt(system_instruction: str, user_prompt: str) -> str:
        """部分只有单 prompt 入口的后端需要把 system + user 合成一段。"""
        if not system_instruction:
            return user_prompt
        return f"System Instruction:\n{system_instruction}\n\nUser Prompt:\n{user_prompt}"


# ==========================================
# 2. OpenAI / GPT (官方 / Azure 兼容)
# ==========================================

class OpenAIClient(BaseLLMClient):
    """OpenAI GPT-4o / GPT-4.1 / o1 等。使用 openai SDK 的 chat.completions 接口。"""

    provider_name = "openai"

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Please install openai: pip install openai") from e
        self._sdk = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url or None,
            timeout=config.timeout,
        )

    def _single_call(self, prompt: str, system_instruction: str) -> str:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        resp = self._sdk.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            **self.config.extra,
        )
        return resp.choices[0].message.content or ""


# ==========================================
# 3. Anthropic / Claude
# ==========================================

class ClaudeClient(BaseLLMClient):
    """Claude 3 / 3.5 / 3.7 / 4 系列，使用 anthropic SDK。"""

    provider_name = "claude"

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("Please install anthropic: pip install anthropic") from e
        self._anthropic = anthropic
        self._sdk = anthropic.Anthropic(
            api_key=config.api_key,
            base_url=config.base_url or None,
            timeout=config.timeout,
        )

    def _single_call(self, prompt: str, system_instruction: str) -> str:
        resp = self._sdk.messages.create(
            model=self.config.model,
            system=system_instruction or None,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
        )
        # resp.content 是一个 list，内含 TextBlock
        parts: List[str] = []
        for block in getattr(resp, "content", []):
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)


# ==========================================
# 4. Google / Gemini
# ==========================================

class GeminiClient(BaseLLMClient):
    """Gemini 1.5 / 2.0 系列，使用 google-generativeai SDK。"""

    provider_name = "gemini"

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install google-generativeai: pip install google-generativeai"
            ) from e
        genai.configure(api_key=config.api_key)
        self._genai = genai
        self._model = genai.GenerativeModel(
            model_name=config.model,
            system_instruction=None,  # 运行时注入
        )

    def _single_call(self, prompt: str, system_instruction: str) -> str:
        # Gemini 需要在 GenerativeModel 级别设 system_instruction；
        # 每次调用都重建一个轻量实例以支持动态 system 消息。
        model = self._genai.GenerativeModel(
            model_name=self.config.model,
            system_instruction=system_instruction or None,
        )
        gen_cfg = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        resp = model.generate_content(prompt, generation_config=gen_cfg)
        return getattr(resp, "text", "") or ""


# ==========================================
# 5. Alibaba Qwen (通义千问) - dashscope compatible-mode
# ==========================================

class QwenClient(BaseLLMClient):
    """
    阿里通义千问。推荐使用 DashScope 兼容 OpenAI 的 endpoint，
    然后复用 openai SDK，统一 chat.completions 调用。
    """

    provider_name = "qwen"
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Please install openai: pip install openai") from e
        self._sdk = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url or self.DEFAULT_BASE_URL,
            timeout=config.timeout,
        )

    def _single_call(self, prompt: str, system_instruction: str) -> str:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        resp = self._sdk.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
        )
        return resp.choices[0].message.content or ""


# ==========================================
# 6. MiniMax
# ==========================================

class MiniMaxClient(BaseLLMClient):
    """
    MiniMax Chat Completions。官方提供 OpenAI 兼容 endpoint:
    https://api.minimax.chat/v1/text/chatcompletion_v2
    这里用 requests 直接 POST 以避免额外 SDK 依赖。
    """

    provider_name = "minimax"
    DEFAULT_BASE_URL = "https://api.minimax.chat/v1/text/chatcompletion_v2"

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import requests  # noqa: F401
        except ImportError as e:
            raise ImportError("Please install requests: pip install requests") from e

    def _single_call(self, prompt: str, system_instruction: str) -> str:
        import requests
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        payload.update(self.config.extra)

        url = self.config.base_url or self.DEFAULT_BASE_URL
        resp = requests.post(url, headers=headers, json=payload,
                             timeout=self.config.timeout)
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return json.dumps(data)


# ==========================================
# 7. DeepSeek
# ==========================================

class DeepSeekClient(BaseLLMClient):
    """
    DeepSeek 官方 API, 兼容 OpenAI 协议。
    默认 base_url: https://api.deepseek.com
    常用模型: deepseek-chat (V3), deepseek-reasoner (R1)
    """

    provider_name = "deepseek"
    DEFAULT_BASE_URL = "https://api.deepseek.com"

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Please install openai: pip install openai") from e
        self._sdk = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url or self.DEFAULT_BASE_URL,
            timeout=config.timeout,
        )

    def _single_call(self, prompt: str, system_instruction: str) -> str:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        # deepseek-reasoner 不支持 temperature / top_p / max_tokens 等部分参数,
        # 这里只在非 reasoner 模型上传这些字段, reasoner 模型交由默认值处理
        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }
        if "reasoner" not in self.config.model.lower():
            kwargs.update({
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
            })
        resp = self._sdk.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


# ==========================================
# 8. Mock (用于测试, 不打网络)
# ==========================================

class MockLLMClient(BaseLLMClient):
    """用于单测/离线场景，返回可配置的固定字符串。"""

    provider_name = "mock"

    def __init__(self, config: Optional[LLMConfig] = None,
                 canned_response: str = "MOCK_RESPONSE"):
        super().__init__(config or LLMConfig(api_key="mock", model="mock"))
        self.canned_response = canned_response

    def _single_call(self, prompt: str, system_instruction: str) -> str:
        return self.canned_response


# ==========================================
# 9. Factory
# ==========================================

class LLMClientFactory:
    """
    通过字符串名字创建任意后端。
    支持从环境变量自动读取 api_key / base_url 作为缺省值。
    """

    _REGISTRY: Dict[str, type] = {
        "openai":   OpenAIClient,
        "gpt":      OpenAIClient,         # alias
        "claude":   ClaudeClient,
        "anthropic": ClaudeClient,         # alias
        "gemini":   GeminiClient,
        "google":   GeminiClient,          # alias
        "qwen":     QwenClient,
        "dashscope": QwenClient,            # alias
        "minimax":  MiniMaxClient,
        "deepseek": DeepSeekClient,
        "mock":     MockLLMClient,
    }

    _ENV_KEY_BY_PROVIDER: Dict[str, str] = {
        "openai":   "OPENAI_API_KEY",
        "gpt":      "OPENAI_API_KEY",
        "claude":   "ANTHROPIC_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini":   "GOOGLE_API_KEY",
        "google":   "GOOGLE_API_KEY",
        "qwen":     "DASHSCOPE_API_KEY",
        "dashscope": "DASHSCOPE_API_KEY",
        "minimax":  "MINIMAX_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }

    @classmethod
    def register(cls, name: str, client_cls: type) -> None:
        """第三方可通过此入口扩展自定义后端。"""
        cls._REGISTRY[name.lower()] = client_cls

    @classmethod
    def available_providers(cls) -> List[str]:
        return sorted(set(cls._REGISTRY.keys()))

    @classmethod
    def create(cls,
               provider: str,
               api_key: Optional[str] = None,
               model: Optional[str] = None,
               base_url: Optional[str] = None,
               temperature: float = 0.7,
               max_tokens: int = 4096,
               top_p: float = 1.0,
               timeout: float = 120.0,
               max_retries: int = 2,
               **extra: Any) -> BaseLLMClient:
        """
        示例::
            create("openai", api_key="...", model="your-model")
            create("openai", model="gpt-4o")  # api_key 从 OPENAI_API_KEY 读取
            create("claude", model="claude-3-5-sonnet-20241022")
        """
        key = provider.lower()
        if key not in cls._REGISTRY:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Available: {cls.available_providers()}"
            )

        # 1) API Key 优先级: 显式参数 > 环境变量
        env_var = cls._ENV_KEY_BY_PROVIDER.get(key)
        resolved_key = api_key or (os.environ.get(env_var) if env_var else None)
        if key != "mock" and not resolved_key:
            raise ValueError(
                f"Provider '{provider}' requires an API key. "
                f"Pass api_key=... or set env var {env_var}."
            )

        # 2) Model 必填（Mock 除外）
        if key != "mock" and not model:
            raise ValueError(f"Provider '{provider}' requires a model / endpoint id.")

        config = LLMConfig(
            api_key=resolved_key or "mock",
            model=model or "mock",
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            timeout=timeout,
            max_retries=max_retries,
            extra=extra or {},
        )
        client_cls = cls._REGISTRY[key]
        return client_cls(config)


# ==========================================
# 10. Self test / CLI
# ==========================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Available providers:", LLMClientFactory.available_providers())

    # Quick mock smoke test
    m = LLMClientFactory.create("mock")
    outs = m.generate("hello", "be terse", n=2)
    assert outs == ["MOCK_RESPONSE", "MOCK_RESPONSE"]
    print("[OK] Mock client works.")
