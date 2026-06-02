#!/usr/bin/env python3
"""Patch evaluate_warmstarts_2d.py to compare raw full-PML warm starts."""

from pathlib import Path


PATH = Path("experiments/2d/evaluate_warmstarts_2d.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"Expected one {label} block, found {count}")
    return text.replace(old, new)


def main() -> None:
    text = PATH.read_text()
    if "base32_raw" in text and "base48_raw" in text:
        print(f"{PATH} already has base32/base48 raw PML methods")
        return

    text = replace_once(
        text,
        '        method_order += ["base32_zero"]\n',
        '        method_order += ["base32_raw", "base32_zero"]\n',
        "base32 method order",
    )
    text = replace_once(
        text,
        '        method_order += ["base48_zero"]\n',
        '        method_order += ["base48_raw", "base48_zero"]\n',
        "base48 method order",
    )
    text = replace_once(
        text,
        '        if "base32" in models:\n'
        '            starts["base32_zero"] = zero_pml_2d(apply_model(models["base32"], u_low, omega_l, cfg, device), cfg)\n',
        '        if "base32" in models:\n'
        '            pred = apply_model(models["base32"], u_low, omega_l, cfg, device)\n'
        '            starts["base32_raw"] = pred\n'
        '            starts["base32_zero"] = zero_pml_2d(pred, cfg)\n',
        "base32 starts",
    )
    text = replace_once(
        text,
        '        if "base48" in models:\n'
        '            starts["base48_zero"] = zero_pml_2d(apply_model(models["base48"], u_low, omega_l, cfg, device), cfg)\n',
        '        if "base48" in models:\n'
        '            pred = apply_model(models["base48"], u_low, omega_l, cfg, device)\n'
        '            starts["base48_raw"] = pred\n'
        '            starts["base48_zero"] = zero_pml_2d(pred, cfg)\n',
        "base48 starts",
    )

    PATH.write_text(text)
    print(f"patched {PATH}")


if __name__ == "__main__":
    main()
