"""Build the Rust core and install it into this Python.

    python -m lab.build_core

Requires a Rust toolchain. If `cargo` is missing, install it from
https://rustup.rs (on Windows: `winget install Rustlang.Rustup`) and re-run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import config

CORE = config.ROOT / "core"
WHEELS = CORE / "target" / "wheels"


def _in_virtualenv() -> bool:
    """`maturin develop` only works inside a venv/conda env; it errors out
    otherwise. On a global Python we build a wheel and pip-install it instead,
    which works everywhere."""
    return bool(os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX"))


def _ensure_cargo_on_path() -> bool:
    """rustup installs to ~/.cargo/bin, which some shells do not pick up until
    they are restarted. Add it ourselves rather than failing on a toolchain
    that is actually installed."""
    if shutil.which("cargo"):
        return True
    cargo_bin = Path.home() / ".cargo" / "bin"
    if (cargo_bin / "cargo.exe").exists() or (cargo_bin / "cargo").exists():
        os.environ["PATH"] = f"{cargo_bin}{os.pathsep}{os.environ['PATH']}"
        return True
    return False


def main() -> int:
    if not _ensure_cargo_on_path():
        print(
            "cargo not found.\n"
            "  Install Rust:  winget install Rustlang.Rustup\n"
            "  then restart the shell and re-run this command."
        )
        return 1
    try:
        import maturin  # noqa: F401
    except ImportError:
        print("installing maturin...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "maturin"])

    print(f"building {CORE} ...")
    if _in_virtualenv():
        subprocess.check_call(
            [sys.executable, "-m", "maturin", "develop", "--release"], cwd=CORE
        )
    else:
        for stale in WHEELS.glob("atomic_core-*.whl"):
            stale.unlink()
        subprocess.check_call(
            [sys.executable, "-m", "maturin", "build", "--release",
             "--interpreter", sys.executable],
            cwd=CORE,
        )
        wheels = sorted(WHEELS.glob("atomic_core-*.whl"))
        if not wheels:
            print(f"build succeeded but no wheel appeared in {WHEELS}")
            return 1
        print(f"\ninstalling {wheels[-1].name} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--force-reinstall",
             "--no-deps", "--quiet", str(wheels[-1])]
        )

    import atomic_core

    board = atomic_core.Board()
    print(f"\nok: {board!r}")
    print(f"    {len(board.legal_moves())} legal moves from startpos")
    print(f"    perft(4) = {atomic_core.perft(atomic_core.STARTPOS, 4):,}")
    print("\nnow verify against Fairy-Stockfish:")
    print("    python -m lab.perft --backend rust --max-depth 5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
