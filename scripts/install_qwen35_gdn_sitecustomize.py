#!/usr/bin/env python3

from __future__ import annotations

import importlib
import site
import sysconfig
from pathlib import Path


MARKER = "# gsdg-qwen35-gdn-sitecustomize"
SITECUSTOMIZE_SNIPPET = f'''{MARKER}
from importlib.util import find_spec


def _gsdg_patch_qwen35_gdn_prefill() -> None:
    if find_spec("flashinfer") is not None:
        return

    try:
        import vllm.model_executor.models.qwen3_next as qwen3_next
    except Exception:
        return

    qwen3_next.ChunkGatedDeltaRule.forward_cuda = (
        qwen3_next.ChunkGatedDeltaRule.forward_native
    )

    current_init = qwen3_next.ChunkGatedDeltaRule.__init__
    if getattr(current_init, "__module__", "") == __name__:
        return

    def _patched_init(self) -> None:
        qwen3_next.CustomOp.__init__(self)
        qwen3_next.logger.warning(
            "flashinfer is not installed; forcing native GDN prefill kernel."
        )
        self._forward_method = self.forward_native

    qwen3_next.ChunkGatedDeltaRule.__init__ = _patched_init


_gsdg_patch_qwen35_gdn_prefill()
'''


def main() -> int:
    candidate_paths: list[Path] = []

    try:
        loaded_sitecustomize = importlib.import_module("sitecustomize")
    except Exception:
        loaded_sitecustomize = None
    else:
        loaded_path = getattr(loaded_sitecustomize, "__file__", "")
        if loaded_path:
            candidate_paths.append(Path(loaded_path))

    candidate_dirs = []
    for key in ("purelib", "platlib"):
        value = sysconfig.get_paths().get(key)
        if value and value not in candidate_dirs:
            candidate_dirs.append(value)

    for directory in candidate_dirs:
        candidate_paths.append(Path(directory) / "sitecustomize.py")

    if site.ENABLE_USER_SITE:
        candidate_paths.append(Path(site.getusersitepackages()) / "usercustomize.py")

    deduped_paths = []
    seen_paths = set()
    for path in candidate_paths:
        normalized = str(path)
        if normalized in seen_paths:
            continue
        seen_paths.add(normalized)
        deduped_paths.append(path)

    if not deduped_paths:
        raise SystemExit("Could not determine a Python customization path")

    last_error = None
    for target in deduped_paths:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            existing = target.read_text() if target.exists() else ""
            if MARKER in existing:
                print(f"sitecustomize patch already present at {target}")
                return 0

            new_text = existing.rstrip()
            if new_text:
                new_text += "\n\n"
            new_text += SITECUSTOMIZE_SNIPPET
            new_text += "\n"
            target.write_text(new_text)
            print(f"Installed Qwen3.5 GDN fallback sitecustomize at {target}")
            return 0
        except OSError as exc:
            last_error = exc

    raise SystemExit(f"Failed to install sitecustomize patch: {last_error}")


if __name__ == "__main__":
    raise SystemExit(main())