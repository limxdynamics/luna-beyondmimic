"""Installation script for the 'whole_body_tracking' python package."""

import os
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "psutil",
]

# Installation operation. This repository is supported as an editable source checkout only. The
# setup metadata exists for ``pip install -e``; a distributable wheel is intentionally unsupported
# because the runtime depends on the large robot assets beside this package.
setup(
    name="whole_body_tracking",
    packages=["whole_body_tracking"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    maintainer_email="opensource@limxdynamics.com",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    # Multi-licensed: unmodified upstream files stay MIT, LimX's own additions/changes are
    # Apache-2.0, and a couple of files keep their own preserved BSD-3-Clause header. See LICENCE,
    # LICENSES/, NOTICE, and THIRD_PARTY_NOTICES.md for the file-level boundaries.
    license="Apache-2.0 AND MIT AND BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Private :: Do Not Upload",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.5.0",
    ],
    zip_safe=False,
)
