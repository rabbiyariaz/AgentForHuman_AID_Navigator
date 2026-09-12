"""Amazon Bedrock AgentCore Runtime entry point for AID Navigator."""

from __future__ import annotations

from typing import Any

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from agent.extraction_agent import run_extraction_pipeline
from triage.ranking import classify_case


app = BedrockAgentCoreApp()


def _request_value(payload: dict[str, Any], name: str) -> str | None:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"'{name}' must be a string.")
    return value


@app.entrypoint
def invoke(payload: dict[str, Any]) -> dict[str, Any]:

    """Process one intake request and return a caseworker decision package."""
    print(f"DEBUG raw payload type: {type(payload)!r}")
    print(f"DEBUG raw payload value: {payload!r}")

    """Process one intake request and return a caseworker decision package."""
    if not isinstance(payload, dict):
        raise ValueError("Request payload must be a JSON object.")

    text = _request_value(payload, "text")
    if not text or not text.strip():
        raise ValueError("'text' is required and must not be empty.")

    document_text = _request_value(payload, "document_text")
    result = run_extraction_pipeline(
        text=text,
        document_path_or_text=document_text,
    )

    response: dict[str, Any] = {
        "case": result,
        "triage": classify_case(result),
    }
    return response


if __name__ == "__main__":
    app.run()