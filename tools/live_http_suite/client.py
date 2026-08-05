"""HTTP client for the live suite: hard timeouts + transport-only retries.

Retry policy (issue #15 harness rules):
- Retry only on 429, 5xx, or connection errors.
- Retry only BEFORE any body/stream bytes are received.
- Never retry semantic assertion failures (caller responsibility).
"""

from __future__ import annotations

import http.client
import json
import socket
import time
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urljoin, urlparse


@dataclass
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: str


class TransportError(Exception):
    """Connection-level failure before a usable complete response."""

    def __init__(self, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause

    def __str__(self) -> str:
        return self.message


def should_retry_transport(
    *,
    status: int | None,
    connection_error: bool,
    body_bytes_received: bool,
) -> bool:
    """Pure policy helper for unit tests and the live client.

    Returns True only when a transport retry is allowed:
    - connection errors before any body bytes
    - 429 / 5xx before any body bytes
    Never after body/stream bytes have started.
    """
    if body_bytes_received:
        return False
    if connection_error:
        return True
    if status is None:
        return False
    return status == 429 or status >= 500


class LiveHttpClient:
    """http.client-based client with hard timeouts and transport retries.

    Status/headers are read before the body so 429/5xx can be retried without
    consuming body bytes. Connect uses connect_timeout_s. Body reads honor a
    wall-clock deadline of timeout_s from the start of the request attempt.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 60.0,
        connect_timeout_s: float = 5.0,
        max_transport_retries: int = 3,
        retry_backoff_s: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_s = timeout_s
        self.connect_timeout_s = connect_timeout_s
        self.max_transport_retries = max_transport_retries
        self.retry_backoff_s = retry_backoff_s
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"unsupported URL scheme: {parsed.scheme!r}")
        self._scheme = parsed.scheme
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self._base_path = parsed.path if parsed.path else "/"

    def url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return urljoin(self.base_url, path.lstrip("/"))

    def _host_header(self) -> str:
        default_port = 443 if self._scheme == "https" else 80
        if self._port == default_port:
            return self._host
        return f"{self._host}:{self._port}"

    def _request_path(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return urlparse(path).path or "/"
        if not path.startswith("/"):
            path = "/" + path
        if self._base_path not in {"", "/"}:
            base = self._base_path if self._base_path.endswith("/") else self._base_path + "/"
            return urljoin(base, path.lstrip("/"))
        return path

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        headers: Mapping[str, str] | None = None,
        retry: bool = True,
    ) -> HttpResponse:
        """Issue one HTTP request. Retries transport failures only when retry=True."""
        req_path = self._request_path(path)
        hdrs = {
            "Accept": "application/json",
            "Host": self._host_header(),
            "Connection": "close",
        }
        data: bytes | None = None
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            hdrs["Content-Type"] = "application/json"
            hdrs["Content-Length"] = str(len(data))
        if headers:
            hdrs.update(headers)

        attempts = self.max_transport_retries if retry else 1
        last_err: BaseException | None = None
        for attempt in range(1, attempts + 1):
            deadline = time.monotonic() + self.timeout_s
            try:
                status, resp_headers, body_reader = self._once_headers(
                    method, req_path, data=data, headers=hdrs, deadline=deadline
                )
            except TransportError as e:
                last_err = e
                if not retry or attempt >= attempts:
                    raise
                if not should_retry_transport(
                    status=None, connection_error=True, body_bytes_received=False
                ):
                    raise
                time.sleep(self.retry_backoff_s * attempt)
                continue

            # Status-based transport retry: abandon body unread, then retry.
            if (
                retry
                and should_retry_transport(
                    status=status, connection_error=False, body_bytes_received=False
                )
                and attempt < attempts
            ):
                body_reader.close()
                time.sleep(self.retry_backoff_s * attempt)
                continue

            # Read body (final attempt or non-retryable status).
            try:
                body = body_reader.read_body(deadline=deadline)
            except TransportError as e:
                last_err = e
                received = body_reader.bytes_received
                body_reader.close()
                if (
                    retry
                    and attempt < attempts
                    and should_retry_transport(
                        status=None,
                        connection_error=True,
                        body_bytes_received=received > 0,
                    )
                ):
                    time.sleep(self.retry_backoff_s * attempt)
                    continue
                raise
            return HttpResponse(status=status, headers=resp_headers, body=body)

        assert last_err is not None
        raise TransportError(str(last_err), cause=last_err)

    def _connect(self) -> http.client.HTTPConnection:
        if self._scheme == "https":
            return http.client.HTTPSConnection(
                self._host, self._port, timeout=self.connect_timeout_s
            )
        return http.client.HTTPConnection(
            self._host, self._port, timeout=self.connect_timeout_s
        )

    def _once_headers(
        self,
        method: str,
        path: str,
        *,
        data: bytes | None,
        headers: Mapping[str, str],
        deadline: float,
    ) -> tuple[int, dict[str, str], "_BodyReader"]:
        conn = self._connect()
        try:
            conn.connect()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TransportError(f"wall-clock timeout before request {method} {path}")
            if conn.sock is not None:
                conn.sock.settimeout(remaining)
            conn.request(method, path, body=data, headers=dict(headers))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TransportError(f"wall-clock timeout awaiting headers {method} {path}")
            if conn.sock is not None:
                conn.sock.settimeout(remaining)
            resp = conn.getresponse()
            status = int(resp.status)
            raw_headers = {k: v for k, v in resp.getheaders()}
            reader = _BodyReader(conn, resp)
            return status, raw_headers, reader
        except TransportError:
            try:
                conn.close()
            except Exception:
                pass
            raise
        except (TimeoutError, socket.timeout) as e:
            try:
                conn.close()
            except Exception:
                pass
            raise TransportError(f"timeout for {method} {path}: {e}", cause=e) from e
        except (ConnectionError, socket.error, OSError, http.client.HTTPException) as e:
            try:
                conn.close()
            except Exception:
                pass
            raise TransportError(f"connection error for {method} {path}: {e}", cause=e) from e

    def get_json(self, path: str) -> HttpResponse:
        return self.request("GET", path)

    def post_json(
        self,
        path: str,
        body: Any,
        *,
        headers: Mapping[str, str] | None = None,
        retry: bool = True,
    ) -> HttpResponse:
        return self.request("POST", path, json_body=body, headers=headers, retry=retry)


class _BodyReader:
    """Lazy body reader so retries can abandon the response before body bytes."""

    def __init__(self, conn: http.client.HTTPConnection, resp: http.client.HTTPResponse) -> None:
        self._conn = conn
        self._resp = resp
        self._closed = False
        self._bytes_received = 0

    @property
    def bytes_received(self) -> int:
        return self._bytes_received

    def read_body(self, *, deadline: float) -> str:
        chunks: list[bytes] = []
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TransportError(
                        f"wall-clock timeout reading body after {self._bytes_received} bytes"
                    )
                if self._conn.sock is not None:
                    self._conn.sock.settimeout(remaining)
                try:
                    chunk = self._resp.read(65536)
                except http.client.IncompleteRead as e:
                    partial = e.partial or b""
                    self._bytes_received += len(partial)
                    if partial:
                        chunks.append(partial)
                    raise TransportError(
                        f"incomplete body after {self._bytes_received} bytes: {e}",
                        cause=e,
                    ) from e
                except (TimeoutError, socket.timeout) as e:
                    raise TransportError(
                        f"timeout reading body after {self._bytes_received} bytes: {e}",
                        cause=e,
                    ) from e
                except (ConnectionError, socket.error, OSError, http.client.HTTPException) as e:
                    raise TransportError(
                        f"connection error reading body after {self._bytes_received} bytes: {e}",
                        cause=e,
                    ) from e
                if not chunk:
                    break
                self._bytes_received += len(chunk)
                chunks.append(chunk)
            raw = b"".join(chunks)
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw.decode("utf-8", errors="replace")
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._resp.close()
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass
