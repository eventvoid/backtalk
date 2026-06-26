#!/usr/bin/env python3
"""Generate a reversed-prompt training file from data/stories.jsonl.

For each line, builds a `training_prompt` whose input-text and story are each
reversed character-by-character, while keeping the original `input` and `story`
untouched.
"""
import json
import sys

SRC = "data/stories.jsonl"
DST = "data/stories_reversed_prompts.jsonl"

# Strict field order for the input text rendering.
INPUT_ORDER = ["hero", "setting", "problem", "helper_or_item", "lesson"]


def input_to_text(inp: dict) -> str:
    return "\n".join(f"{key}: {inp[key]}" for key in INPUT_ORDER)


def reverse(text: str) -> str:
    return text[::-1]


def build_training_prompt(inp: dict, story: str) -> str:
    reversed_input = reverse(input_to_text(inp))
    reversed_story = reverse(story)
    return f"<|input|>\n{reversed_input}\n\n<|story|>\n{reversed_story}\n<|end|>"


def main() -> None:
    n_in = 0
    n_out = 0
    with open(SRC, "r", encoding="utf-8") as fin, \
            open(DST, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            obj = json.loads(line)
            inp = obj["input"]
            story = obj["story"]
            record = {
                "training_prompt": build_training_prompt(inp, story),
                "input": inp,
                "story": story,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Read {n_in} records, wrote {n_out} records to {DST}", file=sys.stderr)


if __name__ == "__main__":
    main()
