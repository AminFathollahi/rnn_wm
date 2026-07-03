"""Render the manuscript figures to vector-format PDFs. Not yet implemented.
"""
from __future__ import annotations
import argparse


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.parse_args(argv)
    raise SystemExit(
        "not yet implemented; this entrypoint exists so the Makefile pipeline resolves."
    )


if __name__ == "__main__":
    main()
