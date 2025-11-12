# Security Policy

## Supported Versions

We actively support the following versions of EdgeFirst HAL with security updates:

| Version | Support Status          |
|---------|------------------------|
| 0.x     | ✅ Full support         |
| < 0.1   | ❌ Not yet released     |

*Note: Version support policy will be updated once we reach stable 1.0 release.*

## Reporting a Vulnerability

Au-Zone Technologies takes security seriously across our entire EdgeFirst ecosystem, including the Hardware Abstraction Layer.

### How to Report

**Email**: support@au-zone.com  
**Subject**: "Security Vulnerability - HAL"

**For GitHub Users**: You can also report vulnerabilities privately through [GitHub Security Advisories](https://github.com/EdgeFirstAI/hal/security/advisories/new)

Please include:
- **Vulnerability description** - Clear explanation of the security issue
- **Steps to reproduce** - Detailed reproduction steps
- **Affected versions** - Which versions are impacted
- **Potential impact** - Assessment of severity and risk
- **Suggested fixes** - If you have ideas for remediation (optional)
- **Proof of concept** - Code or configuration demonstrating the issue (if applicable)

### What to Expect

1. **Acknowledgment** within **48 hours** of your report
2. **Initial assessment** within **7 business days**
3. **Fix timeline** based on severity:
   - **Critical**: 7 days
   - **High**: 30 days
   - **Medium**: Next minor release
   - **Low**: Next major release

### Response Process

Once we receive your report:

1. We'll confirm receipt and begin investigation
2. We'll assess severity using CVSS scoring
3. We'll develop and test a fix
4. We'll coordinate disclosure timing with you
5. We'll release the fix and publish an advisory
6. We'll update affected users through multiple channels

## Responsible Disclosure

We ask that you:

- **Allow reasonable time** for us to fix vulnerabilities before public disclosure
- **Avoid public disclosure** until we've released a patch and advisory
- **Not exploit vulnerabilities** for any purpose other than verification
- **Keep findings confidential** until coordinated disclosure
- **Act in good faith** toward our users and the security community

We commit to:

- **Acknowledge your report** promptly
- **Keep you informed** throughout the remediation process
- **Credit you** in advisories (unless you prefer to remain anonymous)
- **Work collaboratively** to understand and resolve the issue

## Security Update Process

Security updates are released through:

1. **GitHub Security Advisories** - Published on our repository
2. **Release Notes** - Documented in CHANGELOG.md
3. **crates.io Updates** - New versions with security fixes
4. **PyPI Updates** - Python package updates
5. **Email Notifications** - For users who subscribe to security alerts

### Subscribing to Security Updates

- Watch the repository for security advisories on GitHub
- Monitor our [releases page](https://github.com/EdgeFirstAI/hal/releases)
- Follow [@AuZoneTech](https://twitter.com/AuZoneTech) for announcements

## Recognition

With your permission, we'll credit you in:

- **Security advisories** - Public acknowledgment of your contribution
- **Release notes** - Recognition in version release documentation
- **Annual security report** - Listed as a security contributor
- **Hall of fame** - On our website (if you prefer)

If you prefer to remain anonymous, we'll respect that choice.

## Security Best Practices

When using EdgeFirst HAL in production:

### Input Validation
- **Validate all external inputs** before processing
- **Sanitize file paths** and user-provided data
- **Check tensor dimensions** before allocation
- **Verify image formats** match expectations

### Memory Safety
- The HAL is written in Rust, providing memory safety guarantees
- Use safe APIs and avoid unsafe blocks when possible
- Be cautious with FFI boundaries (G2D, OpenGL)
- Monitor memory allocation limits

### Dependency Management
- **Keep dependencies updated** - Regularly update to latest versions
- **Review security advisories** - Check for known vulnerabilities
- **Use cargo audit** - Scan for security issues in dependencies

```bash
cargo install cargo-audit
cargo audit
```

### Hardware Acceleration
- **Validate hardware capabilities** before use
- **Handle fallback paths** securely
- **Limit resource consumption** on shared hardware
- **Isolate untrusted inputs** from accelerators

### Python Bindings
- **Validate numpy arrays** before conversion
- **Check data types** match expectations
- **Handle exceptions** properly
- **Limit memory exposure** across FFI boundary

## Known Security Considerations

### DMA-Heap Memory
- DMA-heap buffers can be shared between processes
- File descriptors can be passed via Unix sockets
- Ensure proper access control on shared buffers
- Consider security implications in multi-tenant environments

### Hardware Accelerators
- G2D and OpenGL access may require elevated privileges
- Shared GPU memory can be a side-channel
- Validate all parameters passed to hardware APIs
- Monitor for resource exhaustion attacks

### Image Processing
- Malformed images can cause decoder crashes
- Large images can exhaust memory
- Implement size limits for untrusted inputs
- Use safe image decoding libraries

## Additional Security Services

For production deployments requiring enhanced security:

### Enterprise Security Offerings

Au-Zone Technologies provides:

- **Security Audits** - Comprehensive code and architecture reviews
- **Penetration Testing** - Third-party security assessments
- **Compliance Certification** - Help meeting regulatory requirements
- **Priority Security Patches** - Expedited fixes for enterprise customers
- **Custom Security Hardening** - Tailored security enhancements
- **Dedicated Security Support** - Direct access to security team

### Professional Services

- **Secure Integration** - Help integrating HAL securely into your stack
- **Threat Modeling** - Assess risks for your specific use case
- **Security Training** - Educate your team on secure usage
- **Incident Response** - Support during security incidents

📧 Contact: support@au-zone.com  
🌐 Learn more: [au-zone.com/security](https://au-zone.com/security)

## Security Disclosure History

We maintain transparency about security issues:

- See [CHANGELOG.md](CHANGELOG.md) for security-related releases
- View published advisories on our [security advisories page](https://github.com/EdgeFirstAI/hal/security/advisories)

*No security vulnerabilities have been disclosed to date.*

---

**Last Updated**: November 2025

Thank you for helping keep EdgeFirst HAL and our users safe! 🔒
