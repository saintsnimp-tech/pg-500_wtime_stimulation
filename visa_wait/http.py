from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import dataclass
from typing import Any


USER_AGENT = (
    "pg-500-wtime-research/1.0 "
    "(+https://github.com/saintsnimp-tech/pg-500_wtime_stimulation)"
)


class CollectionError(RuntimeError):
    """Raised when a public source cannot be collected safely."""


@dataclass(frozen=True)
class HttpPolicy:
    timeout_seconds: float = 30.0
    attempts: int = 3
    min_delay_seconds: float = 0.7
    respect_robots: bool = True


class PoliteJsonClient:
    """Small, dependency-free JSON client with retries, delay and robots checks."""

    def __init__(self, policy: HttpPolicy | None = None) -> None:
        self.policy = policy or HttpPolicy()
        self._last_request_at = 0.0
        self._robots_cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def _robots_allows(self, url: str) -> bool:
        if not self.policy.respect_robots:
            return True
        parsed = urllib.parse.urlsplit(url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        if root not in self._robots_cache:
            robots_url = f"{root}/robots.txt"
            parser = urllib.robotparser.RobotFileParser(robots_url)
            try:
                request = urllib.request.Request(robots_url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=self.policy.timeout_seconds) as response:
                    text = response.read().decode("utf-8", errors="replace")
                parser.parse(text.splitlines())
                self._robots_cache[root] = parser
            except urllib.error.HTTPError as exc:
                # A missing robots file does not create a prohibition.
                self._robots_cache[root] = None if exc.code == 404 else parser
            except (urllib.error.URLError, TimeoutError):
                # Fail closed when a robots policy exists but cannot be checked.
                raise CollectionError(f"Unable to verify robots.txt for {root}")
        parser = self._robots_cache[root]
        return True if parser is None else parser.can_fetch(USER_AGENT, url)

    def get_json(self, url: str) -> dict[str, Any]:
        if not self._robots_allows(url):
            raise CollectionError(f"robots.txt disallows collection: {url}")

        last_error: Exception | None = None
        for attempt in range(self.policy.attempts):
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self.policy.min_delay_seconds:
                time.sleep(self.policy.min_delay_seconds - elapsed)
            request = urllib.request.Request(
                url,
                headers={"Accept": "application/json", "User-Agent": USER_AGENT},
            )
            try:
                self._last_request_at = time.monotonic()
                with urllib.request.urlopen(request, timeout=self.policy.timeout_seconds) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    payload = json.loads(response.read().decode(charset))
                if not isinstance(payload, dict):
                    raise CollectionError(f"Expected a JSON object from {url}")
                return payload
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 < self.policy.attempts:
                    time.sleep((2**attempt) + random.random() * 0.25)
        raise CollectionError(f"Failed after {self.policy.attempts} attempts: {url}") from last_error

