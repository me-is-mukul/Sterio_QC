"""Google Cloud credentials for Cloud Storage and Vertex AI.

Modes (config ``gcp.credentials``):
  adc     Application Default Credentials. On a Compute Engine VM this is the VM's service
          account (no setup); elsewhere `gcloud auth application-default login`.
  gcloud  Short-lived tokens from `gcloud auth print-access-token`. Honours the gcloud
          configuration, including `auth/impersonate_service_account`. Refreshed automatically.
  auto    adc if it is available, else gcloud.

Tokens are never logged or written to disk by this module.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from datetime import datetime, timedelta, timezone

from video_pipeline.core.errors import ConfigError, IngestionError

logger = logging.getLogger(__name__)

CREDENTIAL_MODES = frozenset({"auto", "adc", "gcloud"})
_CLOUD_PLATFORM = "https://www.googleapis.com/auth/cloud-platform"
# gcloud tokens live ~60 min; refresh well before that.
_GCLOUD_TOKEN_LIFETIME = timedelta(minutes=45)


def _gcloud_executable() -> str:
    for name in ("gcloud", "gcloud.cmd"):
        found = shutil.which(name)
        if found:
            return found
    raise IngestionError("credentials mode 'gcloud' needs the gcloud CLI on PATH.")


def _gcloud_token() -> str:
    try:
        proc = subprocess.run([_gcloud_executable(), "auth", "print-access-token"], capture_output=True, text=True,
                              timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise IngestionError(f"Could not run 'gcloud auth print-access-token': {exc}") from exc
    if proc.returncode != 0:
        tail = " | ".join(line for line in proc.stderr.strip().splitlines()[-3:])
        raise IngestionError(f"'gcloud auth print-access-token' failed: {tail}")
    # gcloud may print warnings on stderr; the token is the last non-empty stdout line.
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise IngestionError("'gcloud auth print-access-token' printed no token.")
    return lines[-1]


def _gcloud_credentials():
    from google.oauth2.credentials import Credentials  # noqa: PLC0415

    def refresh_handler(request, scopes):  # noqa: ARG001 (signature required by google-auth)
        expiry = (datetime.now(timezone.utc) + _GCLOUD_TOKEN_LIFETIME).replace(tzinfo=None)
        return _gcloud_token(), expiry

    return Credentials(token=None, refresh_handler=refresh_handler)


def get_credentials(mode: str):
    """Return (credentials, description). The description is safe to log and store."""
    if mode not in CREDENTIAL_MODES:
        raise ConfigError(f"'gcp.credentials' must be one of {sorted(CREDENTIAL_MODES)}, got '{mode}'.")
    if mode in {"auto", "adc"}:
        import google.auth  # noqa: PLC0415
        from google.auth.exceptions import DefaultCredentialsError  # noqa: PLC0415

        try:
            credentials, _project = google.auth.default(scopes=[_CLOUD_PLATFORM])
            account = getattr(credentials, "service_account_email", None) or type(credentials).__name__
            return credentials, f"adc ({account})"
        except DefaultCredentialsError as exc:
            if mode == "adc":
                raise IngestionError(f"Application Default Credentials are not available: {exc}") from exc
            logger.info("No Application Default Credentials; using gcloud access tokens.")
    _gcloud_token()  # fail early with a clear message
    return _gcloud_credentials(), "gcloud access token"
