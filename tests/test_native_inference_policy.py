import pytest

from spatial_mapping_phase2.native_inference_policy import (
    NativeInferencePolicy,
    NativeInferencePolicyError,
)


def test_native_inference_policy_accepts_tested_native_controls() -> None:
    for process_res in (252, 504):
        NativeInferencePolicy(process_res=process_res).validate()


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (NativeInferencePolicy(process_res=768), "measured process resolutions"),
        (NativeInferencePolicy(process_res=252, persistent_model_worker=False), "bounded reuse"),
        (NativeInferencePolicy(process_res=252, cublas_workspace_config=":16:8"), "workspace"),
        (
            NativeInferencePolicy(process_res=252, deterministic_algorithms=False),
            "deterministic PyTorch",
        ),
        (
            NativeInferencePolicy(process_res=252, cudnn_deterministic=False),
            "deterministic PyTorch",
        ),
        (NativeInferencePolicy(process_res=252, cudnn_benchmark=True), "benchmark algorithm"),
        (NativeInferencePolicy(process_res=252, use_upstream_autocast=False), "unmodified source"),
        (
            NativeInferencePolicy(process_res=252, use_upstream_swiglu_fallback=False),
            "unmodified source",
        ),
        (NativeInferencePolicy(process_res=252, enable_xformers=True), "unvalidated acceleration"),
        (
            NativeInferencePolicy(process_res=252, enable_torch_compile=True),
            "unvalidated acceleration",
        ),
        (NativeInferencePolicy(process_res=252, infer_gs=True), "Gaussian inference"),
        (NativeInferencePolicy(process_res=252, export_enabled=True), "Gaussian inference"),
    ],
)
def test_native_inference_policy_rejects_unapproved_controls(
    policy: NativeInferencePolicy, message: str
) -> None:
    with pytest.raises(NativeInferencePolicyError, match=message):
        policy.validate()
