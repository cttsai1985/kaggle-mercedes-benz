"""Create an owner-only Jupyter token; never print its value."""

import os
import secrets
from pathlib import Path

path = Path(".local/jupyter_token")
path.parent.mkdir(exist_ok=True)
Path(".local/runs").mkdir(exist_ok=True)
if path.exists():
    raise SystemExit("Existing Jupyter token retained")
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w") as stream:
    stream.write(secrets.token_urlsafe(32) + "\n")
print("Created owner-only .local/jupyter_token")
