#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def ok(message: str) -> None:
    print(f"[OK] {message}")


def main() -> int:
    root = Path(__file__).resolve().parents[1]

    canonical_rules = root / "configs" / "rules_v1.json"
    service_rules = root / "hanging_mvp" / "rules_mvp.json"

    if not canonical_rules.exists():
        fail("configs/rules_v1.json is missing")

    json.loads(canonical_rules.read_text(encoding="utf-8"))
    ok("canonical rules JSON parses")

    if service_rules.is_symlink():
        target = os.readlink(service_rules)
        expected = "../configs/rules_v1.json"
        if target != expected:
            fail(
                f"hanging_mvp/rules_mvp.json symlink target is {target!r}, "
                f"expected {expected!r}"
            )
        ok("rules_mvp.json points to canonical rules")
    else:
        fail("hanging_mvp/rules_mvp.json must remain a symlink to ../configs/rules_v1.json")

    service_py = root / "hanging_mvp" / "service.py"
    tree = ast.parse(service_py.read_text(encoding="utf-8"))
    async_names = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
    ]
    sync_names = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    ]

    if "analyze_hanging_mvp" not in async_names + sync_names:
        fail("analyze_hanging_mvp endpoint function is missing")

    ok("hanging endpoint function is present")

    py_files = [
        p
        for p in root.rglob("*.py")
        if "__pycache__" not in p.parts
        and ".pytest_cache" not in p.parts
        and ".git" not in p.parts
    ]

    for path in py_files:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")

    ok(f"python syntax compiles for {len(py_files)} files")

    allow_unauth_local = os.getenv("ALLOW_UNAUTH_LOCAL", "").strip().lower() in {"1", "true", "yes", "on"}
    if not os.getenv("API_KEY") and not allow_unauth_local:
        fail("API_KEY is not set; set API_KEY for deployment or ALLOW_UNAUTH_LOCAL=1 for local development")
    if allow_unauth_local:
        print("[WARN] ALLOW_UNAUTH_LOCAL=1 is enabled; do not use this setting for deployment")
    ok("authentication configuration is explicit")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
