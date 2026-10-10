"""Stable identities independent of filename and upload session UUID."""
import hashlib
import json


def fingerprint_files(files):
    parts = sorted((f["suffix"].lower(), f["sha256"]) for f in files)
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()
