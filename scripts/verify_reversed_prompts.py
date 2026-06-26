#!/usr/bin/env python3
"""Validate data/stories_reversed_prompts.jsonl against data/stories.jsonl."""
import json
import sys

SRC = "data/stories.jsonl"
DST = "data/stories_reversed_prompts.jsonl"
INPUT_ORDER = ["hero", "setting", "problem", "helper_or_item", "lesson"]


def input_to_text(inp):
    return "\n".join(f"{key}: {inp[key]}" for key in INPUT_ORDER)


def main():
    errors = 0
    n_src = sum(1 for _ in open(SRC, "r", encoding="utf-8"))
    n_dst = 0

    with open(SRC, "r", encoding="utf-8") as fsrc, \
            open(DST, "r", encoding="utf-8") as fdst:
        for i, (sline, dline) in enumerate(zip(fsrc, fdst), start=1):
            n_dst += 1
            src = json.loads(sline)        # validates source JSON
            try:
                dst = json.loads(dline)    # validates each output line is JSONL
            except json.JSONDecodeError as e:
                print(f"line {i}: invalid JSON: {e}", file=sys.stderr)
                errors += 1
                continue

            # required keys
            for key in ("training_prompt", "input", "story"):
                if key not in dst:
                    print(f"line {i}: missing key {key}", file=sys.stderr)
                    errors += 1

            # input & story unchanged
            if dst.get("input") != src["input"]:
                print(f"line {i}: input mismatch", file=sys.stderr)
                errors += 1
            if dst.get("story") != src["story"]:
                print(f"line {i}: story mismatch", file=sys.stderr)
                errors += 1

            # markers present
            tp = dst.get("training_prompt", "")
            for marker in ("<|input|>", "<|story|>", "<|end|>"):
                if marker not in tp:
                    print(f"line {i}: missing marker {marker}", file=sys.stderr)
                    errors += 1

            # reversal correctness
            expected_input = input_to_text(src["input"])[::-1]
            expected_story = src["story"][::-1]
            expected_tp = (
                f"<|input|>\n{expected_input}\n\n<|story|>\n{expected_story}\n<|end|>"
            )
            if tp != expected_tp:
                print(f"line {i}: training_prompt content mismatch", file=sys.stderr)
                errors += 1

    # count check (covers any length difference between the two files)
    n_dst_total = sum(1 for _ in open(DST, "r", encoding="utf-8"))
    print(f"source lines: {n_src}")
    print(f"output lines: {n_dst_total}")
    print(f"compared lines: {n_dst}")
    if n_src != n_dst_total:
        print("COUNT MISMATCH", file=sys.stderr)
        errors += 1

    if errors:
        print(f"FAILED with {errors} error(s)", file=sys.stderr)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
