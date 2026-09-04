# Security policy

## Supported versions

Before the first tagged public release, security fixes are made on the current default branch.
There is no support commitment for old commits, private training runs, locally modified robot
descriptions, or deployment stacks outside this repository.

## Reporting a vulnerability

Do **not** open a public issue for an unpatched vulnerability.

Report vulnerabilities privately to:

**opensource@limxdynamics.com**

Include:

- the affected commit, file, model, motion, or configuration;
- impact and the conditions required to reproduce it;
- minimal reproduction steps or a proof of concept;
- whether real hardware, people, customer/field data, or credentials may be affected;
- a safe contact method for follow-up.

Do not send live credentials, private keys, customer data, raw field logs, or other sensitive
payloads in the first message. The maintainers will arrange an appropriate transfer method if the
material is required.

The maintainers will acknowledge and triage reports as soon as practical. No fixed response or
remediation SLA is currently promised. Please coordinate public disclosure with the maintainers so
users have a reasonable opportunity to apply a fix.

## Scope

Security reports may cover:

- source code and dependency handling;
- unsafe deserialization or untrusted motion/model inputs;
- bundled model, motion, or robot-description artifacts;
- ONNX metadata and deployment-interface validation;
- behavior that could cause unsafe commands or unexpected actuation on supported robot hardware.

An ONNX signature, metadata, simulation, or unit-test pass does not establish hardware safety.
Site-specific risk assessment, emergency handling, fall protection, and operator procedures remain
the deployer's responsibility.

This project does not currently operate a public bug-bounty program.

## Accepted trusted-input risk

The retargeted `.npy` preparation path uses NumPy pickle loading before schema validation. A
malicious file can therefore execute code under the user's account.

On 2026-09-01, the project owner accepted this residual risk and decided not to change the input
format. The supported boundary is limited to `.npy` files generated or verified inside LimX
Dynamics. Files downloaded from the internet, received from an untrusted third party, or otherwise
of unknown provenance are unsupported and must not be loaded.

The warning in the English and Chinese README and Quick Start is part of the security control and
must not be removed while the pickle input format remains supported.
