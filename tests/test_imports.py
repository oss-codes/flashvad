from __future__ import annotations

import subprocess
import sys


def test_native_runtime_import_does_not_eagerly_load_torch() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import flashvad.native_runtime; "
                "assert 'torch' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
