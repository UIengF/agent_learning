from __future__ import annotations

from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlparse

from .config import PROJECT_ROOT, PermissionConfig


class PermissionDenied(ValueError):
    pass


@dataclass(frozen=True)
class ToolPermissionPolicy:
    config: PermissionConfig
    project_root: Path = PROJECT_ROOT

    def _allowed_index_roots(self) -> tuple[Path, ...]:
        roots = []
        raw_roots = [item.strip() for item in self.config.allowed_index_roots.split(";")]
        for raw_root in raw_roots:
            if not raw_root:
                continue
            path = Path(raw_root)
            if not path.is_absolute():
                path = self.project_root / path
            roots.append(path.resolve())
        return tuple(roots)

    def validate_index_dir(self, index_dir: str | Path) -> str:
        candidate = Path(index_dir)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        resolved = candidate.resolve()

        allowed_roots = self._allowed_index_roots()
        if not allowed_roots:
            return str(index_dir)

        for root in allowed_roots:
            if resolved == root or root in resolved.parents:
                return str(index_dir)
        raise PermissionDenied(
            f"index_dir is outside allowed roots: {resolved}. "
            f"Allowed roots: {', '.join(str(root) for root in allowed_roots)}"
        )

    def validate_web_fetch_url(self, url: str) -> str:
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"}:
            raise PermissionDenied("web_fetch only allows http and https URLs.")
        host = (parsed.hostname or "").lower()
        if not host:
            raise PermissionDenied("web_fetch URL must include a host.")
        if not self.config.allow_local_web_fetch and host in {"localhost"}:
            raise PermissionDenied("web_fetch to localhost is disabled by policy.")
        if not self.config.allow_private_web_fetch:
            try:
                address = ip_address(host)
            except ValueError:
                address = None
            if address and (address.is_private or address.is_loopback or address.is_link_local):
                raise PermissionDenied(
                    "web_fetch to private or loopback IPs is disabled by policy."
                )
        return url.strip()


def build_permission_policy(config: PermissionConfig) -> ToolPermissionPolicy:
    return ToolPermissionPolicy(config=config)
