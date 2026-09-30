# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for a security vulnerability. Instead,
report it privately via GitHub's responsible-disclosure flow (Security →
"Report a vulnerability") or by contacting the repository maintainers directly.

Please include:

- A description of the vulnerability and the affected component
- Steps to reproduce (if safe to share)
- Suggested remediation, if known

You'll receive an acknowledgement, and we aim to triage within a few days.

## Supported versions

Security fixes are applied to the current `main` branch and the latest tagged
release.

## Security notes for this project

- **Never commit credentials.** Repository `.env` files are git-ignored and
  contain only placeholders. Copy `.env.example` or `docs/deploy.env.example`
  and set real values per host.
- Rotate any credential exposed historically before production use.
- Production deployments should terminate TLS at a reverse proxy and restrict
  `CORS_ORIGINS` to the real origin.
- API keys for live data sources (NASA FIRMS, Copernicus CDS) should be supplied
  via environment variables, never hard-coded.
