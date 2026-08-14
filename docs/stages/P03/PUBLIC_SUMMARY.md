# P03 Public Summary - Live capture and temporal-gating foundation

P03 added immutable stream-profile, frame, capture-session, artifact and selected-bundle contracts,
plus bounded reconnect-safe capture through a shared service used by both CLI and localhost web
operations. It preserves source timing, host acquisition timing, fallback provenance, hashing,
partial/failure states and profile incompatibility without exposing configured endpoints.

The accepted foundation deliberately separates bundle completeness from synchronization. A bundle
may contain every configured camera while still being temporally unsuitable for a downstream
consumer. The post-acceptance revision adds warm decoded-frame rings and a caller-supplied temporal
gate in the host-monotonic acquisition domain. Its conservative bound includes clock quantization;
passing it does not prove simultaneous sensor exposure or make independent source clocks
comparable.

Private RTSP endpoints, credentials, media, manifests and retained live artifacts are excluded.
The public tests use synthetic adapters and cover reconnects, timeouts, cancellation, backpressure,
fallback provenance, immutable replay, temporal rejection and service/UI parity.
