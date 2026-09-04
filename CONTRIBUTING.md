# Contributing

Thank you for contributing to the Luna (HU_L04) motion-tracking project.

## Before opening an issue

- Search existing issues and documentation.
- Use the security process in [`SECURITY.md`](SECURITY.md) for vulnerabilities or unsafe
  real-hardware behavior; do not disclose an unpatched issue publicly.
- Do not attach credentials, customer/field logs, raw captures, private robot configuration, or
  other sensitive material to a public issue.

## Development setup

Follow [`README.md`](README.md) or [`README.zh-CN.md`](README.zh-CN.md) for the supported
Isaac Sim 4.5 / Isaac Lab 2.1 / Python 3.10 environment.

Install the repository and development checks:

```bash
python -m pip install -e source/whole_body_tracking
python -m pip install pre-commit
pre-commit install
```

Only this editable source installation is supported. Do not publish the package to PyPI or treat a
locally built wheel/sdist as a supported artifact; the runtime depends on source-tree robot assets
that are intentionally outside the package artifact.

## Making a change

1. Create a focused branch from the current default branch.
2. Keep functional changes, governance/docs changes, and generated artifacts in separate commits
   when practical.
3. Preserve existing copyright and SPDX headers.
4. Update documentation when changing a public command, file format, observation layout, action
   mapping, or deployment contract.
5. Add or update tests that fail before the change and pass after it.

### Changes that require maintainer approval before implementation

- adding, removing, reordering, or resizing policy observations or actions;
- changing robot-control behavior, actuator gains, limits, termination semantics, deployment
  metadata, or executable examples;
- changing dependencies, installation behavior, public APIs, supported file formats, or existing
  test meaning;
- adding or changing robot assets, motion data, checkpoints, ONNX files, binaries, or generated
  artifacts;
- choosing or changing a license, copyright statement, redistribution rule, or sensitive-data
  boundary.

## Data, model, and asset contributions

Do not submit a motion, model, robot description, mesh, binary, screenshot, log, or other
non-source artifact without:

- a fixed source identity and checksum;
- owner/collector and modification history;
- license or approved use/redistribution terms;
- privacy, customer/field, and hardware-sensitive boundary review where applicable;
- required attribution, data card, or model card.

This repository's code is multi-licensed — unmodified upstream files stay MIT, LimX Dynamics'
additions and changes are Apache-2.0, and a couple of files keep their own preserved BSD-3-Clause
header — and none of that automatically applies to bundled assets, data, or models. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the current boundaries.

## Validation

Run the checks relevant to your change:

```bash
pre-commit run --all-files
python -m unittest discover -s tests -v
python tests/test_local_only_workflow.py -v
python tests/test_combined_export_npz_motion.py -v
```

The full test collection must pass. Include the exact commands and results in the pull request;
targeted tests do not replace the full suite.

GPU/Isaac Sim and real-hardware validation must be reported separately from CPU-only tests. An
environment limitation is not a pass.

## Pull requests

Use a concise title and explain:

- why the change is needed;
- files and behavior affected;
- validation run and results;
- compatibility or migration impact;
- data/model/license and robot-safety implications, or state explicitly that there are none.

By submitting source code, you agree that your contribution may be distributed under the
applicable license already assigned to the files you modify. Contributions involving separately
licensed assets, data, or models require explicit maintainer approval and may use different terms.
