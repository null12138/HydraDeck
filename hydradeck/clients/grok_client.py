from __future__ import annotations

import json
import time
from dataclasses import dataclass

import requests

from hydradeck.utils import Heartbeat

JSON = dict[str, object]

CHROME_144_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/144.0.0.0 Safari/537.36"
)


class GrokClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class GrokClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 180.0,
        max_retries: int = 3,
        heartbeat: bool = False,
        heartbeat_interval_s: float = 5.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._heartbeat = heartbeat
        self._heartbeat_interval_s = heartbeat_interval_s

    def chat_text(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.3,
        timeout_s: float | None = None,
    ) -> str:
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        data = self._post_chat(
            {"model": self._model, "messages": msgs, "temperature": temperature},
            timeout_s=timeout_s,
        )
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise GrokClientError(f"No choices in response: {data}")
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, str):
            raise GrokClientError(f"No message.content in response: {data}")
        return content.strip()

    def chat_json(
        self,
        messages: list[ChatMessage],
        schema_hint: str,
        temperature: float = 0.2,
        timeout_s: float | None = None,
    ) -> JSON:
        suffix = (
            "\n\nReturn ONLY valid JSON. Do not include markdown fences. "
            "If unsure, still return best-effort JSON that matches: "
            + schema_hint
        )
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        if msgs and msgs[-1].get("role") == "user":
            msgs[-1]["content"] = str(msgs[-1]["content"]) + suffix
        else:
            msgs.append({"role": "user", "content": suffix})

        text = self.chat_text(
            [ChatMessage(role=m["role"], content=m["content"]) for m in msgs],
            temperature=temperature,
            timeout_s=timeout_s,
        )
        parsed = _best_effort_json_parse(text)
        if parsed is None:
            raise GrokClientError("Model did not return valid JSON. Response was:\n" + text)
        return parsed

    def _post_chat(self, payload: JSON, timeout_s: float | None = None) -> JSON:
        url = f"{self._base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json", "User-Agent": CHROME_144_UA}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        effective_timeout = float(timeout_s) if timeout_s is not None else self._timeout_s

        last_err: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                with Heartbeat(
                    enabled=self._heartbeat,
                    label=f"POST {url}",
                    interval_s=self._heartbeat_interval_s,
                ):
                    r = requests.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=effective_timeout,
                    )
                if r.status_code >= 400:
                    raise GrokClientError(f"HTTP {r.status_code} from {url}: {r.text[:2000]}")
                data = r.json()
                if not isinstance(data, dict):
                    raise GrokClientError("Non-object response")
                return data
            except (requests.RequestException, ValueError, GrokClientError) as e:
                last_err = e
                if attempt >= self._max_retries:
                    break
                time.sleep(0.5 * (2**attempt))
        raise GrokClientError(f"Request failed after retries: {last_err}")

    def list_models(self, timeout_s: float | None = None) -> list[str]:
        url = f"{self._base_url}/v1/models"
        headers: dict[str, str] = {"User-Agent": CHROME_144_UA}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        effective_timeout = float(timeout_s) if timeout_s is not None else self._timeout_s
        with Heartbeat(
            enabled=self._heartbeat,
            label=f"GET {url}",
            interval_s=self._heartbeat_interval_s,
        ):
            r = requests.get(url, headers=headers, timeout=effective_timeout)
        if r.status_code >= 400:
            raise GrokClientError(f"HTTP {r.status_code} from {url}: {r.text[:2000]}")
        data = r.json()
        if not isinstance(data, dict):
            raise GrokClientError("Non-object response")
        raw = data.get("data")
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                mid = item.get("id")
                if isinstance(mid, str):
                    out.append(mid)
        return out


class MockClient:
    def chat_text(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        timeout_s: float | None = None,
    ) -> str:
        _ = temperature
        _ = timeout_s
        joined = "\n".join([f"{m.role}: {m.content}" for m in messages])
        low = joined.lower()
        if "write a detailed pre-research report" in low:
            return "\n".join(
                [
                    "# Pre-Research Report",
                    "",
                    "## Research questions",
                    "- (Mock) What is the core problem?",
                    "",
                    "## Scope & non-scope",
                    "- Scope: offline mock run",
                    "- Non-scope: real web browsing",
                    "",
                    "## Search plan & queries",
                    "- query 1",
                    "- query 2",
                    "",
                    "## Risks & limitations",
                    "- Mock output is not evidence-backed",
                    "",
                ]
            )
        if "write a long-form research report" in low:
            return (
                "# Research Report\n\n"
                "## Summary\n(Mock)\n\n"
                "## Resources\n1. Example Source 1 — https://example.com\n"
            )
        if "speech script" in low:
            return (
                "# Speech Script\n\n"
                "## Opening\n(Mock)\n\n"
                "## Main\n(Mock)\n\n"
                "## Closing\n(Mock)\n"
            )
        if "critique the current research plan" in low:
            return "- (Mock) Missing primary sources\n- (Mock) Claims need evidence\n"
        if "sources" in joined.lower():
            return json.dumps(
                {
                    "sources": [
                        {
                            "url": "https://example.com",
                            "title": "Example Source 1",
                            "snippet": "Mock source for offline run.",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        if "facts" in joined.lower():
            return json.dumps(
                {
                    "facts": [
                        {
                            "claim": "Mock mode produces deterministic artifacts.",
                            "evidence": "MockClient returns fixed outputs.",
                            "url": "https://example.com",
                            "title": "Example Source 1",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        if "outline" in joined.lower():
            return json.dumps(
                {
                    "outline": [
                        "Background",
                        "Problem formulation",
                        "Methods",
                        "Findings",
                        "Limitations",
                        "Open questions",
                    ]
                },
                ensure_ascii=False,
            )
        return "Mock synthesis text."

    def chat_json(
        self,
        messages: list[ChatMessage],
        schema_hint: str,
        temperature: float = 0.0,
        timeout_s: float | None = None,
    ) -> JSON:
        _ = schema_hint
        _ = timeout_s
        joined = "\n".join([f"{m.role}: {m.content}" for m in messages])
        low = joined.lower()
        if "score" in low and "rubric" in low and "return json" in low:
            return {
                "score": 0.99,
                "reasons": ["mock pass"],
                "must_fix": [],
            }
        if "pre_report_md" in low and "paper_tex" in low and "slides_tex" in low:
            return {
                "pre_report_md": "\n".join(
                    [
                        "# Pre-Research (Mock)",
                        "",
                        "## 15-minute agenda",
                        "- 0:00-2:00 Background",
                        "- 2:00-6:00 Research questions",
                        "- 6:00-10:00 Evidence plan",
                        "- 10:00-13:00 Risks",
                        "- 13:00-15:00 Deliverables",
                        "",
                        "## Research questions",
                        "- RQ1 ...",
                        "- RQ2 ...",
                        "",
                        "## Search plan & queries",
                        "- query 1",
                        "- query 2",
                        "",
                        "## Resources",
                        "1. Example Source 1 — https://example.com",
                        "",
                    ]
                ),
                "report_md": "\n".join(
                    [
                        "# Research Report (Mock)",
                        "",
                        "## Summary",
                        "(Mock)",
                        "",
                        "## Findings",
                        "- (Mock) claim with [1]",
                        "",
                        "## Resources",
                        "[1] Example Source 1 — https://example.com",
                        "",
                    ]
                ),
                "speech_md": "\n".join(
                    [
                        "# Speech (Mock)",
                        "",
                        "[0:00] Opening hook",
                        "[2:00] Transition",
                        "[8:00] Key point",
                        "[14:00] Close + Q&A",
                        "",
                    ]
                ),
                "paper_tex": "\\documentclass{article}\\n\\begin{document}Mock\\end{document}\\n",
                "slides_tex": "\\documentclass{beamer}\\n\\begin{document}Mock\\end{document}\\n",
                "bibtex": "@misc{src1,title={Example},howpublished={\\url{https://example.com}}}\n",
            }

        text = self.chat_text(messages, temperature=temperature)
        parsed = _best_effort_json_parse(text)
        return parsed or {"ok": True}


def _best_effort_json_parse(text: str) -> JSON | None:
    t = text.strip()
    if not t:
        return None
    if t.startswith("{") and t.endswith("}"):
        try:
            v = json.loads(t)
            if isinstance(v, dict):
                return v
        except Exception:
            pass

    start = t.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = t[start : i + 1]
                try:
                    v2 = json.loads(chunk)
                    if isinstance(v2, dict):
                        return v2
                except Exception:
                    return None
    return None
