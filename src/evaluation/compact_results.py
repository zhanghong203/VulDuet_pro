"""Convert an existing full result file into the compact result schema."""

import argparse
import json
from pathlib import Path

from src.util.compact_result import compact_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result")
    parser.add_argument("--output")
    args = parser.parse_args()

    source = Path(args.result)
    destination = Path(args.output) if args.output else source.with_name(
        source.stem.replace("_result", "") + "_compact_result.json"
    )
    rows = json.loads(source.read_text(encoding="utf-8"))
    compact = [compact_pair(row) for row in rows]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "source": str(source.resolve()),
        "output": str(destination.resolve()),
        "pairs": len(compact),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
