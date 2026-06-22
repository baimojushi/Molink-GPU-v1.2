#!/usr/bin/env python3
"""Parse artwork physical dimensions from a filename pattern like
   'artwork51.2×67.8cm.png' → width=0.512, height=0.678 (metres).

Supported separators between the two numbers: ×, x, X, *, ·, ·
Supported unit suffixes (case-insensitive): cm, mm, m, or absent (=cm).
If unit is absent or 'cm', values are divided by 100 to produce metres.
If unit is 'mm', values are divided by 1000.
If unit is 'm', values are used directly.
Any other unit suffix causes an error.

Output: prints ARTWORK_WIDTH_M and ARTWORK_HEIGHT_M as shell-export lines.
"""

import re
import sys
import os


def parse_artwork_dimensions(filename: str):
    """Return (width_m, height_m) from a filename string."""
    basename = os.path.basename(filename)
    # strip extension
    name = os.path.splitext(basename)[0]

    # pattern: capture first number, separator, second number, optional unit
    # numbers may contain decimal point
    pat = re.compile(
        r'(\d+(?:\.\d+)?)'       # first number (width)
        r'\s*'                     # optional whitespace
        r'[×xX\*·\.x]'            # separator (multiplication sign variants)
        r'\s*'                     # optional whitespace
        r'(\d+(?:\.\d+)?)'       # second number (height)
        r'\s*'                     # optional whitespace
        r'([a-zA-Z]*)'            # optional unit suffix
        r'$',
    )

    m = pat.search(name)
    if m is None:
        raise ValueError(f"Cannot extract dimensions from filename: {basename}")

    w_val = float(m.group(1))
    h_val = float(m.group(2))
    unit = m.group(3).strip().lower()

    if unit == '' or unit == 'cm':
        scale = 0.01
    elif unit == 'mm':
        scale = 0.001
    elif unit == 'm':
        scale = 1.0
    else:
        raise ValueError(
            f"Unsupported unit '{unit}' in filename: {basename}. "
            "Expected 'cm', 'mm', 'm', or no unit (assumes cm)."
        )

    return w_val * scale, h_val * scale


def main():
    if len(sys.argv) < 2:
        print("Usage: parse_artwork_dimensions.py <filename_or_path>", file=sys.stderr)
        sys.exit(1)

    filename = sys.argv[1]
    w_m, h_m = parse_artwork_dimensions(filename)

    print(f"ARTWORK_WIDTH_M={w_m:.6f}")
    print(f"ARTWORK_HEIGHT_M={h_m:.6f}")


if __name__ == "__main__":
    main()
