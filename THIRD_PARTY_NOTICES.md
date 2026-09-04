# Third-party notices and license boundaries

This file records source and license boundaries for content distributed in this repository. It is
informational and does not replace or modify the referenced license terms.

## Modified upstream codebase

This repository is a modified fork of:

- **Project:** HybridRobotics `whole_body_tracking` / BeyondMimic motion tracking
- **Source:** <https://github.com/HybridRobotics/whole_body_tracking>
- **Fixed baseline:** `cd65172032893724b445448818c34165846d847d`
  (`Add observation history lengths to ONNX metadata`, 2025-10-05)
- **License:** MIT
- **License text:** [`LICENCE`](LICENCE)
- **Modification state:** modified-vendored. This repository is published as a single commit rather
  than as the upstream history plus patches, so the modifications are established by diffing this
  tree against the fixed baseline in the upstream repository above, and are summarized in
  [`CHANGELOG.md`](CHANGELOG.md).

The upstream MIT copyright and permission notice are preserved verbatim in the top-level
`LICENCE`. The original copyright statement must not be removed when redistributing modified
versions. At the fixed baseline above, the public upstream's own `LICENCE` contains that same Isaac
Lab Project Developers notice and does not contain a separate HybridRobotics copyright line. This
notice records the upstream project and source without inventing a copyright statement that the
upstream artifact does not make.

## LimX Dynamics modifications

Everything LimX Dynamics added or changed relative to that baseline is licensed under Apache-2.0:

- **License text:** [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt)
- **Attribution notice:** [`NOTICE`](NOTICE)

MIT and Apache-2.0 are compatible, and both permit the combined work. Apache-2.0 additionally
carries an express patent grant and a requirement to state that files have been changed; the
statement for this repository is the fork record above plus [`CHANGELOG.md`](CHANGELOG.md), which
lists the changes made relative to the baseline. A recipient of the
combined work must satisfy both licenses, which in practice means retaining `LICENCE`,
`LICENSES/Apache-2.0.txt` and `NOTICE`.

Where a file carries its own license header, that header governs the file and takes precedence
over this section. Those files are listed below.

## BSD-3-Clause source files

The following modified files retain copyright and SPDX headers from the Isaac Lab Project
Developers and are distributed under BSD-3-Clause:

- `scripts/rsl_rl/train.py`
- `source/whole_body_tracking/whole_body_tracking/utils/exporter.py`

Their complete license terms are in [`LICENSES/BSD-3-Clause.txt`](LICENSES/BSD-3-Clause.txt).
The file headers remain the authoritative copyright notices for each file.

## Configuration snippets

`.gitattributes` retains comments identifying configuration copied or adapted from:

- [NVIDIA Warp](https://github.com/NVIDIA/warp), Apache-2.0, copyright NVIDIA Corporation &
  Affiliates.
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab), BSD-3-Clause, copyright the Isaac Lab
  Project Developers.

Both license texts are in `LICENSES/`.

## Community governance template

`CODE_OF_CONDUCT.md` is adapted from the
[Contributor Covenant, version 2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct.html),
maintained by the Organization for Ethical Source and licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The project-specific enforcement contact
is a local modification.

## Bundled motion data

`motions/*.npy` is LimX Dynamics' own material and is covered by the Apache-2.0 grant above, with
no separate terms. Its provenance, processing history and review status are recorded in
[`motions/README.md`](motions/README.md).

## Robot description

The Luna robot description is **not** distributed by this repository. It is referenced as a Git
submodule at `robot_descriptions/luna-description`, pointing at
<https://github.com/limx-luna/luna-description>, and is licensed Apache-2.0 by that repository.
Its license and notice files travel with it; nothing in this repository restates or supersedes
them.

## External dependencies

Packages installed by pip, conda, Isaac Sim, or Isaac Lab are external dependencies and are not
bundled by this source repository. Their own upstream licenses apply. Dependency names and versions
are described in `README.md`, `Install.md`, `source/whole_body_tracking/setup.py`, and
`tools/offline_onnx_exporter/requirements.txt`.
