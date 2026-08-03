#!/usr/bin/env python3
"""Minimal JSON-over-stdio simulator used by the README example."""

from __future__ import annotations

import json
import sys


def main() -> None:
    request = json.load(sys.stdin)
    if request.get("tool") != "compound_growth":
        raise ValueError("unknown tool")
    arguments = request.get("arguments", {})
    initial = float(arguments["initial_value"])
    rate = float(arguments["rate_percent"])
    years = int(arguments["years"])
    if initial < 0 or not -100 <= rate <= 1000 or not 0 <= years <= 100:
        raise ValueError("argument outside handbook range")
    final = initial * (1 + rate / 100) ** years
    print(
        json.dumps(
            {
                "final_value": final,
                "absolute_change": final - initial,
                "percent_change": (final / initial - 1) * 100 if initial else 0.0,
            }
        )
    )


if __name__ == "__main__":
    main()
