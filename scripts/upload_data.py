"""Upload content-addressed CI data to Cloudflare R2 via Wrangler OAuth.

Usage (from a Pixi task):
    pixi run --locked -e r2-upload upload-data path/to/file.tar.gz

Prerequisites:
    - project-local Wrangler declared in package.json/package-lock.json
    - nodejs and Python managed by Pixi
    - `wrangler login` completed for the user (one time):

    pixi run --locked -e r2-upload login-r2
"""

import argparse
import configparser
import hashlib
import json
import mimetypes
import os
import pathlib
import shutil
import subprocess
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

R2_BUCKET = "ci-data"
BASE_URL = "https://data.openfe.org"
CACHE_CONTROL = "public, max-age=31536000, immutable"
API_BASE = "https://api.cloudflare.com/client/v4"
# Conservative ceiling for Wrangler/Cloudflare REST single-object uploads.
# Use an S3-compatible client for larger files.
WRANGLER_MAX_UPLOAD_BYTES = 300_000_000
RCLONE_REMOTE = "openfe-r2"

# Make sure we are using utils managed by pixi
# This file needs to be placed in the repo root in a folder
# i.e. scripts/upload_data.py
NPM_ROOT = pathlib.Path(__file__).resolve().parent / "r2-upload"

def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wrangler(*args: str, account_id: str | None = None) -> str:
    """Run the pinned, project-local Wrangler and capture its stdout."""
    env = os.environ.copy()
    if account_id is not None:
        # Select the discovered account for this subprocess only.
        env["CLOUDFLARE_ACCOUNT_ID"] = account_id

    result = subprocess.run(
        ["npm", "exec", "--", "wrangler", *args],
        cwd=NPM_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def ensure_wrangler() -> None:
    """Install the pinned npm dependencies in the isolated tool directory."""
    if not (NPM_ROOT / "package-lock.json").is_file():
        raise RuntimeError(
            "Missing scripts/r2-upload/package-lock.json. "
            "Initialize the Wrangler npm project before uploading."
        )

    if not (NPM_ROOT / "node_modules/.bin/wrangler").is_file():
        subprocess.run(
            ["npm", "ci", "--no-audit", "--no-fund"],
            cwd=NPM_ROOT,
            check=True,
        )


def authentication_headers() -> dict[str, str]:
    """Reuse Wrangler credentials without printing or persisting secrets."""
    auth = json.loads(wrangler("auth", "token", "--json"))
    match auth.get("type"):
        case "oauth" | "api_token":
            return {"Authorization": f"Bearer {auth['token']}"}
        case "api_key":
            return {"X-Auth-Key": auth["key"], "X-Auth-Email": auth["email"]}
        case _:
            raise RuntimeError("Wrangler has no usable credentials; run `pixi run --locked npm exec -- wrangler login`.")


def cf_get(
    path: str,
    headers: dict[str, str],
    *,
    params: dict[str, str | int] | None = None,
    missing_ok: bool = False,
) -> dict | None:
    url = f"{API_BASE}{path}"
    if params:
        url += "?" + urlencode(params)

    try:
        with urlopen(Request(url, headers=headers), timeout=30) as response:
            payload = json.load(response)
    except HTTPError as exc:
        if missing_ok and exc.code == 404:
            return None
        raise RuntimeError(f"Cloudflare API request failed (HTTP {exc.code}): {path}") from exc

    if payload.get("success") is not True:
        raise RuntimeError(f"Cloudflare API error for {path}: {payload.get('errors')}")
    return payload


def discover_bucket_account(headers: dict[str, str]) -> tuple[str, str]:
    """Find the account containing this R2 bucket."""
    identity = json.loads(wrangler("whoami", "--json"))
    accounts = identity.get("accounts")
    if not isinstance(accounts, list) or not accounts:
        raise RuntimeError("Wrangler returned no accessible accounts; check login and permissions.")

    matches: list[tuple[str, str]] = []
    for account in accounts:
        account_id = account["id"]
        path = f"/accounts/{quote(account_id, safe='')}/r2/buckets/{quote(R2_BUCKET, safe='')}"
        payload = cf_get(path, headers, missing_ok=True)
        if payload is None:
            continue
        bucket = payload["result"]
        if bucket["name"] != R2_BUCKET:
            raise RuntimeError(f"Unexpected bucket returned for account {account_id}")
        matches.append((account_id, account.get("name", account_id)))

    if not matches:
        raise RuntimeError(f"Bucket {R2_BUCKET!r} was not found in accessible accounts.")
    if len(matches) > 1:
        names = ", ".join(name for _, name in matches)
        raise RuntimeError(f"Bucket {R2_BUCKET!r} is ambiguous across accounts: {names}")
    return matches[0]


def existing_objects(
    account_id: str, digest: str, headers: dict[str, str]
) -> list[dict]:
    """List all files under v1/<sha256>/, regardless of filename."""
    path = (
        f"/accounts/{quote(account_id, safe='')}/r2/buckets/"
        f"{quote(R2_BUCKET, safe='')}/objects"
    )
    objects: list[dict] = []
    cursor = None
    while True:
        params: dict[str, str | int] = {
            "prefix": f"v1/{digest}/",
            "per_page": 1000,
        }
        if cursor:
            params["cursor"] = cursor

        payload = cf_get(path, headers, params=params)
        assert payload is not None
        objects.extend(payload["result"])
        page = payload.get("result_info") or {}
        if not page.get("is_truncated"):
            return objects
        cursor = page.get("cursor")
        if not cursor:
            raise RuntimeError("R2 returned a truncated listing without a cursor.")



def rclone_s3_flags(account_id: str) -> list[str]:
    """Pin the destination account despite rclone environment overrides."""
    return [
        "--s3-endpoint", f"https://{account_id}.r2.cloudflarestorage.com",
        "--s3-provider", "Cloudflare",
        "--s3-region", "auto",
        "--s3-no-check-bucket",
    ]


def rclone_ready(account_id: str) -> tuple[bool, str]:
    """Validate a named R2 remote without reading its stored secrets."""
    if shutil.which("rclone") is None:
        return False, "rclone is not installed in the Pixi r2-upload environment"

    try:
        proc = subprocess.run(
            ["rclone", "config", "redacted", RCLONE_REMOTE],
            check=True,
            text=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=20,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return False, f"rclone configuration is unavailable ({type(exc).__name__})"

    config = configparser.ConfigParser(interpolation=None)
    try:
        config.read_string(proc.stdout)
    except configparser.Error:
        return False, "rclone returned an unparseable configuration"
    if not config.has_section(RCLONE_REMOTE):
        return False, f"rclone remote {RCLONE_REMOTE!r} is not configured"

    remote = config[RCLONE_REMOTE]
    expected_endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    if remote.get("type") != "s3" or remote.get("provider", "").lower() != "cloudflare":
        return False, "rclone remote is not a Cloudflare S3 remote"
    if remote.get("endpoint", "").rstrip("/") != expected_endpoint:
        return False, f"rclone endpoint does not match the Cloudflare account owning {R2_BUCKET}"

    # A remote configured for this endpoint may still have expired or
    # insufficient credentials; probe the exact destination bucket read-only.
    try:
        subprocess.run(
            [
                "rclone", "lsf", f"{RCLONE_REMOTE}:{R2_BUCKET}",
                "--max-depth", "1", *rclone_s3_flags(account_id),
            ],
            check=True,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            timeout=40,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return False, f"rclone cannot list the destination bucket ({type(exc).__name__})"
    return True, ""


def upload_with_rclone(
    file: pathlib.Path, key: str, account_id: str, content_type: str
) -> None:
    """Upload using a verified remote; never replace an existing object."""
    subprocess.run(
        [
            "rclone", "copyto", str(file), f"{RCLONE_REMOTE}:{R2_BUCKET}/{key}",
            "--immutable", "--progress",
            "--metadata-set", f"cache-control={CACHE_CONTROL}",
            "--metadata-set", f"content-type={content_type}",
            *rclone_s3_flags(account_id),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=pathlib.Path)
    args = parser.parse_args()

    path = args.file.expanduser().resolve()
    if not path.is_file():
        parser.error(f"Not a file: {path}")

    digest = sha256_file(path)
    requested_key = f"v1/{digest}/{path.name}"

    ensure_wrangler()
    headers = authentication_headers()
    account_id, account_name = discover_bucket_account(headers)
    print(f"Using R2 bucket {R2_BUCKET!r} in Cloudflare account {account_name!r}.")

    matches = existing_objects(account_id, digest, headers)
    if len(matches) > 1:
        keys = ", ".join(item["key"] for item in matches)
        raise RuntimeError(f"Multiple objects have the same SHA256 prefix: {keys}")

    if matches:
        existing = matches[0]
        if existing["size"] != path.stat().st_size:
            raise RuntimeError("Existing object has the expected hash in its key but a different size.")
        key = existing["key"]
        print("Already uploaded; skipping.")
        if key != requested_key:
            print(f"Existing content uses a different filename: {key}")
    else:
        key = requested_key
        content_type = (
            "application/gzip"
            if path.name.endswith((".tar.gz", ".tgz", ".gz"))
            else mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        )
        size = path.stat().st_size
        if size > WRANGLER_MAX_UPLOAD_BYTES:
            ready, reason = rclone_ready(account_id)
            if ready:
                print(f"File exceeds Wrangler's limit ({size:,} bytes); using rclone.")
                upload_with_rclone(path, key, account_id, content_type)
            else:
                url = f"{BASE_URL}/{quote(key, safe='/')}"
                print(f"File is too large for Wrangler ({size:,} bytes).")
                print(f"rclone unavailable: {reason}. No upload was attempted.")
                print("Configure a bucket-scoped rclone remote named 'openfe-r2'.")
                print(f"Endpoint: https://{account_id}.r2.cloudflarestorage.com")
                print(f"Bucket: {R2_BUCKET}")
                print(f"Object key: {key}")
                print(f"SHA256: {digest}")
                print(f"Content-Type: {content_type}")
                print(f"Cache-Control: {CACHE_CONTROL}")
                print(f"URL after manual upload: {url}")
                print(f'Pooch registry: "{path.name}": "sha256:{digest}"')
                print(f'Pooch urls: "{path.name}": "{url}"')
                raise SystemExit(2)
        else:
            print(f"Uploading {path} to {R2_BUCKET}/{key} ...")
            wrangler(
                "r2", "object", "put", f"{R2_BUCKET}/{key}",
                "--file", str(path), "--remote",
                "--content-type", content_type,
                "--cache-control", CACHE_CONTROL,
                account_id=account_id,
            )
        print("Uploaded new object.")

    url = f"{BASE_URL}/{quote(key, safe='/')}"
    print(f"URL: {url}")
    print(f'Pooch registry: "{path.name}": "sha256:{digest}"')
    print(f'Pooch urls: "{path.name}": "{url}"')


if __name__ == "__main__":
    main()
