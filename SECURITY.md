# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of Forge seriously. If you believe you have found a security vulnerability, please report it to us as described below.

**Please do not report security vulnerabilities through public GitHub issues.**

### How to Report

1. **Email**: Send details to security@forge.dev (or open a private security advisory on GitHub)
2. **GitHub Security Advisories**: Use [GitHub's private vulnerability reporting](https://github.com/forge-features/forge/security/advisories/new)

### What to Include

Please include the following information in your report:

- Type of vulnerability (e.g., buffer overflow, SQL injection, cross-site scripting)
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### Response Timeline

- **Initial Response**: Within 48 hours, we will acknowledge receipt of your report
- **Status Update**: Within 7 days, we will provide an initial assessment
- **Resolution**: We aim to resolve critical vulnerabilities within 30 days

### Disclosure Policy

- We will work with you to understand and resolve the issue quickly
- We will keep you informed of our progress
- We will credit you in the security advisory (unless you prefer to remain anonymous)
- We ask that you give us reasonable time to address the issue before public disclosure

## Security Measures

### Automated Security Scanning

Forge uses several automated tools to detect vulnerabilities:

- **Dependabot**: Automated dependency updates for security patches
- **CodeQL**: Static analysis for potential security issues
- **pip-audit**: Python dependency vulnerability scanning in CI
- **SBOM Generation**: Software Bill of Materials generated for each release

### Secure Development Practices

- All code changes require review before merging
- CI/CD pipeline includes security checks
- Dependencies are regularly updated
- We follow OWASP guidelines for secure coding

## Security-Related Configuration

When using Forge in production, consider the following:

1. **Input Validation**: Forge validates input data types but does not sanitize data content
2. **File Paths**: Be cautious when loading data from user-provided file paths
3. **Serialization**: Only load pickled models from trusted sources (use `joblib.load()` carefully)
4. **Dependencies**: Keep Forge and its dependencies updated

## Past Security Advisories

No security advisories have been issued yet.

---

Thank you for helping keep Forge and its users safe!
