#!/usr/bin/env python3
# trim_last_newline.py
import sys, os
from pathlib import Path

def trim_trailing_newlines(p: Path) -> int:
    """
    Remove trailing newline(s) at EOF.
    Returns how many bytes were removed (0, 1, or 2 ... possibly more if multiple newlines).
    """
    removed = 0
    with p.open('rb+') as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size == 0:
            return 0

        # Repeatedly trim CRLF or LF at the very end until none remain
        while size > 0:
            back = 2 if size >= 2 else 1
            f.seek(size - back, os.SEEK_SET)
            tail = f.read(back)

            if tail.endswith(b'\r\n'):
                size -= 2
                removed += 2
                f.truncate(size)
            elif tail.endswith(b'\n') or tail.endswith(b'\r'):
                size -= 1
                removed += 1
                f.truncate(size)
            else:
                break
    return removed

def main():
    # Directory and extensions
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("synthetic_hu")
    exts = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else (".txt",)

    if not root.exists():
        print(f"Directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    changed = 0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in exts:
            try:
                removed = trim_trailing_newlines(p)
                if removed:
                    print(f"trimmed {removed} byte(s): {p}")
                    changed += 1
            except Exception as e:
                print(f"error on {p}: {e}", file=sys.stderr)

    print(f"Done. Files modified: {changed}")

if __name__ == "__main__":
    main()
