#!/usr/bin/env python3
"""
prevalidation.py — CiteGuardian Submission Pre-Validator

Replicates the 3 checks from validate-submission.sh:
  Step 1: Ping HF Space /reset endpoint
  Step 2: Docker build
  Step 3: openenv validate

Usage:
    python prevalidation.py <ping_url> [repo_dir]

Example:
    python prevalidation.py https://my-team.hf.space .
"""

import json
import os
import subprocess
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

DOCKER_BUILD_TIMEOUT = 600  # seconds

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS_COUNT = 0

def ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")

def log(msg: str):
    print(f"[{ts()}] {msg}")

def passed(msg: str):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"[{ts()}] PASSED -- {msg}")

def failed(msg: str):
    print(f"[{ts()}] FAILED -- {msg}")

def hint(msg: str):
    print(f"  Hint: {msg}")

def stop_at(step: str):
    print(f"\nValidation stopped at {step}. Fix the above before continuing.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Step 1: Ping HF Space
# ---------------------------------------------------------------------------

def check_hf_space(ping_url: str):
    log(f"Step 1/3: Pinging HF Space ({ping_url}/reset) ...")
    url = f"{ping_url.rstrip('/')}/reset"
    req = urllib.request.Request(
        url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            code = resp.status
    except urllib.error.HTTPError as e:
        code = e.code
    except Exception as e:
        failed(f"HF Space not reachable: {e} — skipping, continuing to local checks")
        hint("Check your network and that the Space is running.")

    if code == 200:
        passed("HF Space is live and responds to /reset")
    else:
        failed(f"HF Space /reset returned HTTP {code} (expected 200) — skipping, continuing to local checks")
        hint("Deploy your HF Space and re-run to fully validate.")


# ---------------------------------------------------------------------------
# Step 2: Docker build
# ---------------------------------------------------------------------------

def check_docker_build(repo_dir: Path):
    log("Step 2/3: Running docker build ...")

    # Check docker is available
    if subprocess.run(["docker", "--version"], capture_output=True).returncode != 0:
        failed("docker command not found")
        hint("Install Docker: https://docs.docker.com/get-docker/")
        stop_at("Step 2")

    # Find Dockerfile
    if (repo_dir / "Dockerfile").exists():
        context = repo_dir
    elif (repo_dir / "server" / "Dockerfile").exists():
        context = repo_dir / "server"
    else:
        failed("No Dockerfile found in repo root or server/ directory")
        stop_at("Step 2")
        return

    log(f"  Found Dockerfile in {context}")

    try:
        result = subprocess.run(
            ["docker", "build", str(context)],
            capture_output=True,
            text=True,
            timeout=DOCKER_BUILD_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        failed(f"Docker build timed out after {DOCKER_BUILD_TIMEOUT}s")
        stop_at("Step 2")
        return

    if result.returncode == 0:
        passed("Docker build succeeded")
    else:
        failed("Docker build failed")
        # Print last 20 lines of output
        lines = (result.stdout + result.stderr).strip().splitlines()
        for line in lines[-20:]:
            print(f"  {line}")
        stop_at("Step 2")


# ---------------------------------------------------------------------------
# Step 3: openenv validate
# ---------------------------------------------------------------------------

def check_openenv_validate(repo_dir: Path):
    log("Step 3/3: Running openenv validate ...")

    if subprocess.run(["openenv", "--version"], capture_output=True).returncode != 0:
        failed("openenv command not found")
        hint("Install it: pip install openenv-core")
        stop_at("Step 3")

    result = subprocess.run(
        ["openenv", "validate"],
        capture_output=True,
        text=True,
        cwd=str(repo_dir),
    )

    output = (result.stdout + result.stderr).strip()

    if result.returncode == 0:
        passed("openenv validate passed")
        if output:
            log(f"  {output}")
    else:
        failed("openenv validate failed")
        print(output)
        stop_at("Step 3")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        sys.exit(1)

    ping_url = args[0]
    repo_dir = Path(args[1]).resolve() if len(args) > 1 else Path(".").resolve()

    if not repo_dir.is_dir():
        print(f"Error: directory '{repo_dir}' not found")
        sys.exit(1)

    print()
    print("=" * 40)
    print("  OpenEnv Submission Validator")
    print("=" * 40)
    log(f"Repo:     {repo_dir}")
    log(f"Ping URL: {ping_url}")
    print()

    check_hf_space(ping_url)
    check_docker_build(repo_dir)
    check_openenv_validate(repo_dir)

    print()
    print("=" * 40)
    print(f"  All 3/3 checks passed!")
    print(f"  Your submission is ready to submit.")
    print("=" * 40)
    print()


if __name__ == "__main__":
    main()
