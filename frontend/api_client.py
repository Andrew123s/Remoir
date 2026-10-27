"""Thin HTTP client for the Remoir FastAPI backend."""

from __future__ import annotations

import os

import httpx

API_URL = os.environ.get("REMOIR_API_URL", "http://127.0.0.1:8000").rstrip("/")


class APIError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _client(timeout: float = 300) -> httpx.Client:
    headers = {}
    if os.environ.get("REMOIR_API_KEY"):
        headers["X-API-Key"] = os.environ["REMOIR_API_KEY"]
    return httpx.Client(base_url=API_URL, timeout=timeout, headers=headers)


def _handle(resp: httpx.Response):
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        if isinstance(detail, list):
            detail = "; ".join(str(d.get("msg", d)) for d in detail)
        raise APIError(resp.status_code, str(detail))
    if resp.headers.get("content-type", "").startswith("application/json"):
        return resp.json()
    return resp.content


def get(path: str, **params):
    with _client() as c:
        return _handle(c.get(path, params=params or None))


def post(path: str, json: dict | None = None, files=None):
    with _client() as c:
        return _handle(c.post(path, json=json, files=files))


def put(path: str, json: dict):
    with _client() as c:
        return _handle(c.put(path, json=json))


def delete(path: str):
    with _client() as c:
        return _handle(c.delete(path))


def health() -> dict | None:
    try:
        with _client(timeout=5) as c:
            return _handle(c.get("/health"))
    except (httpx.HTTPError, APIError):
        return None
