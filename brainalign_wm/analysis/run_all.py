"""Run the full alignment + stats pipeline over completed runs.

TODO(sonnet5): implement per protocol §9-§10.
"""
from __future__ import annotations
import argparse


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.parse_args(argv)
    raise SystemExit(
        "[stub] not implemented yet — see protocol §9-§10. "
        "This target exists so the Makefile pipeline resolves."
    )


if __name__ == "__main__":
    main()
