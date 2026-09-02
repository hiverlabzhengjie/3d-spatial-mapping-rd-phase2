# Configuration

Store non-secret, versioned project policies and examples here. Runtime configuration must be
validated by typed project models before use. RTSP credentials and private endpoints belong in
the untracked `.env` or local secret store.

Configuration files must declare units, coordinate frames, camera/stream profile versions and
the policy version that interprets them.

Use `console-profile.example.json` as the secret-free template for the maintained combined console.
Copy it to an ignored local filename, replace machine paths, and keep credentials only in the
referenced `.env`. Validate the complete external runtime without starting anything:

```powershell
phase2-console --profile configs/console-profile.local.json --preflight
```

The source-checkout compatibility command remains
`python scripts/run_p08_workflow_console.py ...`.

When XR02 Live operations are enabled, also copy
`xr02-worker-deployment.example.json` to an ignored local filename. Replace every path and recorded
SHA-256 with the exact accepted local input/model identities. The file references `.env` but never
contains endpoint values. The combined `--preflight` verifies all six immutable hashes, the local
four-camera endpoint roster, policy database, runtime paths and external tools before startup.
