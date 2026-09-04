"""Where the robot description lives.

The Luna USD is not vendored here. It comes from the ``luna-description`` submodule, which is
checked out at the repository root rather than inside the Python package: a submodule cannot live
under a path the top-level ``.gitignore`` excludes, and ``assets/`` is excluded wholesale.

Set ``LUNA_DESCRIPTION_DIR`` to point somewhere else - an installed wheel has no repository root
above it, so that is the only way to run this package from outside a checkout.
"""

import os

# Conveniences to other module directories via relative paths
ASSET_DIR = os.path.abspath(os.path.dirname(__file__))

# assets -> whole_body_tracking -> whole_body_tracking -> source -> repository root
_REPO_ROOT = os.path.abspath(os.path.join(ASSET_DIR, os.pardir, os.pardir, os.pardir, os.pardir))

LUNA_DESCRIPTION_DIR = os.environ.get(
    "LUNA_DESCRIPTION_DIR",
    os.path.join(_REPO_ROOT, "robot_descriptions", "luna-description"),
)


def luna_description_path(*parts: str) -> str:
    """Resolve a path inside the description package, failing where the cause is still visible.

    An uninitialised submodule leaves an empty directory behind, so the import succeeds and the
    error only surfaces later as a USD stage that will not open. Check it here instead.
    """
    path = os.path.join(LUNA_DESCRIPTION_DIR, *parts)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} does not exist.\n"
            "The robot description is a submodule and is not part of this repository's checkout by"
            " default. Fetch it with:\n\n"
            "    git submodule update --init robot_descriptions/luna-description\n\n"
            "Or point LUNA_DESCRIPTION_DIR at an existing copy of luna-description."
        )
    return path
