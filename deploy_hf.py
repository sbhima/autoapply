#!/usr/bin/env python3
"""Create the Hugging Face Docker Space and push this repo using a one-time hidden prompt (never saved to disk)."""
from __future__ import annotations

import getpass
import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

SPACE_NAME = os.environ.get("HF_SPACE_NAME", "autoapply")


def get_token() -> str:
    return getpass.getpass("Paste your Hugging Face Write Token (hidden, not saved to disk): ").strip()


def main() -> int:
    token = get_token()
    if not token:
        print("Error: No Hugging Face token provided.")
        return 1

    req = urllib.request.Request(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        who = json.loads(r.read().decode())
    username = who["name"]
    print(f"Authenticated as: {username}")

    create_body = json.dumps({
        "type": "space",
        "name": SPACE_NAME,
        "sdk": "docker",
        "private": False,
    }).encode()
    req_create = urllib.request.Request(
        "https://huggingface.co/api/repos/create",
        data=create_body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req_create, timeout=20) as r:
            res = json.loads(r.read().decode())
            print("Created Space:", res.get("url"))
    except urllib.error.HTTPError as e:
        if e.code == 409:
            print(f"Space {username}/{SPACE_NAME} already exists — pushing update.")
        else:
            raise RuntimeError(f"Failed to create Space ({e.code}): {e.read().decode()}") from e

    repo_root = Path(__file__).parent.resolve()
    clean_url = f"https://huggingface.co/spaces/{username}/{SPACE_NAME}"
    authed_url = f"https://{username}:{token}@huggingface.co/spaces/{username}/{SPACE_NAME}"

    subprocess.run(["git", "remote", "remove", "space"], cwd=repo_root, check=False, capture_output=True)
    subprocess.run(["git", "remote", "add", "space", clean_url], cwd=repo_root, check=True)
    subprocess.run(["git", "push", "--force", authed_url, "main:main"], cwd=repo_root, check=True)

    slug_user = username.lower().replace("_", "-")
    slug_space = SPACE_NAME.lower().replace("_", "-")
    print("\nDeployment pushed!")
    print(f"  • Space Page:     {clean_url}")
    print(f"  • Direct App URL: https://{slug_user}-{slug_space}.hf.space")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
