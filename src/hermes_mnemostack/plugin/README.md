# mnemostack (compatibility shim)

This directory is what hermes-agent **0.19** scans for; it does not contain the
provider. The implementation lives in the `hermes-mnemostack` package on PyPI.

Do not copy it by hand — `hermes-mnemostack install` puts it in the right place
for the *active* Hermes profile and verifies that hermes can actually load it:

```bash
pip install hermes-mnemostack
hermes-mnemostack install
hermes memory setup mnemostack
```

On hermes-agent **0.20 and later** the pip entry point is discovered on its own
and this directory is unnecessary (harmless if present — a directory plugin
takes precedence over an entry point, and both resolve to the same provider).

See the project README for configuration, and SECURITY.md for what leaves the
machine.
