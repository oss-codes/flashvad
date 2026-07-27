# Security policy

FlashVAD is an alpha research project. Only the latest release receives
security fixes.

Please report suspected vulnerabilities through GitHub private vulnerability
reporting. Do not open a public issue containing an exploit, private audio,
credentials, or personally identifiable call data.

Checkpoint loading uses PyTorch's restricted `weights_only` mode. Treat model
files, audio, manifests, and benchmark inputs as untrusted data and validate
their provenance before use.

The browser demo processes audio locally and does not upload recordings.
Production integrators remain responsible for transport authentication,
authorization, consent, retention, and regional privacy requirements.
