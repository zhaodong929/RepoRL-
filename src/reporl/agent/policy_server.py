"""Authenticated single-worker HTTP server for trace-preserving GPU generation."""

from __future__ import annotations

import argparse
import importlib
import os
import threading
from collections.abc import Sequence
from typing import Any

from pydantic import Field

from reporl.agent.hf_policy import TransformersPolicy
from reporl.agent.models import ChatMessage, PolicyIdentity, PolicyStep
from reporl.agent.policy import PolicyBackend
from reporl.schemas import StrictModel


class PolicyServerRequest(StrictModel):
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    seed: int


class PolicyServerInfo(StrictModel):
    status: str = "ok"
    policy_id: str
    policy_revision: str
    policy_identity: PolicyIdentity | None = None


class PolicyServerResponse(StrictModel):
    policy_id: str
    policy_revision: str
    adapter_sha256: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    step: PolicyStep


def execute_request(
    policy: PolicyBackend,
    request: PolicyServerRequest,
    *,
    lock: threading.Lock,
) -> PolicyStep:
    with lock:
        return policy.act(request.messages, seed=request.seed)


def create_app(policy: PolicyBackend, *, bearer_token: str) -> Any:
    if not bearer_token:
        raise ValueError("bearer_token must not be empty")
    try:
        fastapi: Any = importlib.import_module("fastapi")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "policy server requires RepoRL's 'server' optional dependencies"
        ) from error
    app = fastapi.FastAPI(title="RepoRL Trace Policy Server", docs_url=None, redoc_url=None)
    lock = threading.Lock()

    def authorize(value: str | None) -> None:
        if value != f"Bearer {bearer_token}":
            raise fastapi.HTTPException(status_code=401, detail="unauthorized")

    @app.get("/health", response_model=PolicyServerInfo)  # type: ignore[untyped-decorator]
    def health(
        authorization: str | None = fastapi.Header(default=None),
    ) -> PolicyServerInfo:
        authorize(authorization)
        return PolicyServerInfo(
            policy_id=policy.policy_id,
            policy_revision=policy.policy_revision,
            policy_identity=getattr(policy, "policy_identity", None),
        )

    @app.post(  # type: ignore[untyped-decorator]
        "/action",
        response_model=PolicyServerResponse,
    )
    def action(
        request: PolicyServerRequest,
        authorization: str | None = fastapi.Header(default=None),
    ) -> PolicyServerResponse:
        authorize(authorization)
        return PolicyServerResponse(
            policy_id=policy.policy_id,
            policy_revision=policy.policy_revision,
            adapter_sha256=getattr(policy, "adapter_sha256", None),
            step=execute_request(policy, request, lock=lock),
        )

    return app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--adapter")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--max-input-tokens", type=int, default=16_384)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args(argv)
    token = os.environ.get("REPORL_POLICY_SERVER_TOKEN", "")
    if not token:
        parser.error("REPORL_POLICY_SERVER_TOKEN is required")
    policy = TransformersPolicy.from_pretrained(
        args.model,
        revision=args.revision,
        adapter_path=args.adapter,
        load_in_4bit=not args.no_4bit,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    app = create_app(policy, bearer_token=token)
    try:
        uvicorn: Any = importlib.import_module("uvicorn")
    except ModuleNotFoundError as error:
        raise RuntimeError("policy server requires Uvicorn") from error
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
