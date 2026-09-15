"""Install the MAES EP4 import hooks from any composing sitecustomize."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys


_INSTALLERS = {
    "vllm.model_executor.layers.fused_moe.layer": "install_into",
    "vllm.model_executor.models.qwen3_moe": "install_qwen_moe_into",
    "vllm.model_executor.models.qwen3_vl_moe": "install_qwen_vl_loader_into",
}


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def create_module(self, spec):
        method = getattr(self._wrapped, "create_module", None)
        return method(spec) if method is not None else None

    def exec_module(self, module):
        self._wrapped.exec_module(module)
        import src.vllm_ep4_runtime as runtime

        getattr(runtime, _INSTALLERS[module.__name__])(module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _INSTALLERS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _PatchLoader(spec.loader)
        return spec


if os.environ.get("MAES_EP4_PLAN") and not any(
    isinstance(finder, _PatchFinder) for finder in sys.meta_path
):
    sys.meta_path.insert(0, _PatchFinder())
