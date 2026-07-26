"""
Adapters that let your EXISTING agent stand in for the "agent under test"
without reimplementing anything about it here. Both adapters expose the
same minimal interface the rest of the harness expects:

    engine.generate(prompt: str, system: str = "") -> str

Two ways to plug your agent in:

1. ObjectAgentAdapter -- you already have an agent object/instance in this
   process. Point it at whatever method actually runs it.

2. EndpointAgentAdapter -- your agent runs behind an HTTP API (your own
   service, or someone else's). Point it at the URL and, if the request/
   response shape isn't the OpenAI-style default, supply your own
   formatter/parser functions.

Neither adapter cares what's inside your agent -- multi-step tool use,
its own retries, whatever. It just needs to take a question in and
return the final text (e.g. the SQL string) out.
"""
from typing import Callable, Optional


class ObjectAgentAdapter:
    """
    Wraps an existing agent object already living in this process.

    Example:
        from my_project.agent import AnalyticsAgent
        my_agent = AnalyticsAgent(...)
        engine = ObjectAgentAdapter(my_agent, method="run")

    If your agent's call signature or return shape doesn't match the
    defaults below, pass `call` and/or `extract_text` to override.
    """

    def __init__(
        self,
        agent,
        method: str = "run",
        call: Optional[Callable[[object, str, str], object]] = None,
        extract_text: Optional[Callable[[object], str]] = None,
    ):
        self.agent = agent
        self.method = method
        # default call convention: agent.<method>(prompt) -- system prompt
        # prepended if your agent doesn't take one separately
        self.call = call or self._default_call
        # default: assume the return value is already a string, or has
        # a `.content` / `.text` / `.output` attribute, or is a dict with
        # one of those keys
        self.extract_text = extract_text or self._default_extract

    def _default_call(self, agent, prompt: str, system: str):
        fn = getattr(agent, self.method)
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        return fn(full_prompt)

    @staticmethod
    def _default_extract(result) -> str:
        if isinstance(result, str):
            return result
        for attr in ("content", "text", "output", "answer", "sql"):
            if hasattr(result, attr):
                return getattr(result, attr)
            if isinstance(result, dict) and attr in result:
                return result[attr]
        return str(result)

    def generate(self, prompt: str, system: str = "") -> str:
        result = self.call(self.agent, prompt, system)
        return self.extract_text(result).strip()


class EndpointAgentAdapter:
    """
    Wraps an existing agent that runs behind an HTTP API.

    Example (OpenAI-compatible chat endpoint, the default assumption):
        engine = EndpointAgentAdapter(
            url="https://my-service.internal/v1/agent/query",
            headers={"Authorization": "Bearer ..."},
        )

    Example (custom request/response shape):
        engine = EndpointAgentAdapter(
            url="https://my-service.internal/analytics-agent",
            request_formatter=lambda prompt, system: {"query": prompt, "context": system},
            response_parser=lambda resp_json: resp_json["result"]["sql"],
        )
    """

    def __init__(
        self,
        url: str,
        headers: Optional[dict] = None,
        request_formatter: Optional[Callable[[str, str], dict]] = None,
        response_parser: Optional[Callable[[dict], str]] = None,
        timeout: int = 60,
    ):
        import requests  # lazy import so this file doesn't hard-require it
        self._requests = requests
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self.request_formatter = request_formatter or self._default_request
        self.response_parser = response_parser or self._default_parse

    @staticmethod
    def _default_request(prompt: str, system: str) -> dict:
        return {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
        }

    @staticmethod
    def _default_parse(resp_json: dict) -> str:
        # tries a few common response shapes; override with response_parser
        # if your endpoint's schema is different
        if "output" in resp_json:
            return resp_json["output"]
        if "response" in resp_json:
            return resp_json["response"]
        try:
            return resp_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ValueError(
                f"Don't know how to parse this response, pass response_parser=: {resp_json}"
            )

    def generate(self, prompt: str, system: str = "") -> str:
        payload = self.request_formatter(prompt, system)
        resp = self._requests.post(self.url, json=payload, headers=self.headers, timeout=self.timeout)
        resp.raise_for_status()
        return self.response_parser(resp.json()).strip()
