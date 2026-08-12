"""Validation for the narrowly allowed P00 native DA3 runtime controls."""

from __future__ import annotations

from dataclasses import dataclass


class NativeInferencePolicyError(ValueError):
    """Raised when a requested P00 native inference control is not approved."""


@dataclass(frozen=True)
class NativeInferencePolicy:
    """Source-preserving controls validated by P00 for the selected native candidate."""

    process_res: int
    persistent_model_worker: bool = True
    cublas_workspace_config: str = ":4096:8"
    deterministic_algorithms: bool = True
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    use_upstream_autocast: bool = True
    use_upstream_swiglu_fallback: bool = True
    enable_xformers: bool = False
    enable_torch_compile: bool = False
    infer_gs: bool = False
    export_enabled: bool = False

    def validate(self) -> None:
        if self.process_res not in (252, 504):
            raise NativeInferencePolicyError(
                "P00 has only measured process resolutions 252 and 504."
            )
        if not self.persistent_model_worker:
            raise NativeInferencePolicyError(
                "P00's native operational policy requires bounded reuse of one "
                "loaded model worker."
            )
        if self.cublas_workspace_config != ":4096:8":
            raise NativeInferencePolicyError(
                "P00's validated CUDA deterministic workspace is ':4096:8'."
            )
        if not self.deterministic_algorithms or not self.cudnn_deterministic:
            raise NativeInferencePolicyError(
                "P00 requires deterministic PyTorch and cuDNN controls."
            )
        if self.cudnn_benchmark:
            raise NativeInferencePolicyError("P00 rejects cuDNN benchmark algorithm selection.")
        if not self.use_upstream_autocast or not self.use_upstream_swiglu_fallback:
            raise NativeInferencePolicyError(
                "P00 must retain DA3's unmodified source runtime choices."
            )
        if self.enable_xformers or self.enable_torch_compile:
            raise NativeInferencePolicyError(
                "P00 does not authorize unvalidated acceleration implementations."
            )
        if self.infer_gs or self.export_enabled:
            raise NativeInferencePolicyError(
                "The P00 optimization probe does not enable Gaussian inference or exports."
            )

