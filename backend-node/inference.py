"""BackTalk inference engine for a backend node.

Loads the reversed-text models and runs generation. Reuses the model definition
and tokenizer from ../train; the worker provides the gateway transport.
"""
import os
import random
import re
import sys
from typing import Dict, List, Optional

# Some PyTorch operators are not implemented by every macOS/PyTorch
# combination. Let those operators fall back to CPU instead of killing the
# first MPS generation. Operators supported by Metal still run on MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "train"))

from data import BOS_TOKEN, EOS_TOKEN, load_tokenizer, reverse_text  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from train import pick_device, set_seed  # noqa: E402

KIND_ASSISTANT = "assistant"
KIND_STORYTELLER = "storyteller"
END_MARKER = "<|end|>"
DEFAULT_TOKENIZER = os.path.join(ROOT, "tokenizers", "backtalk-tokenizer", "tokenizer.json")

INPUT_ORDER = ["hero", "setting", "problem", "helper_or_item", "lesson"]
STORY_OPTIONS = {
    "hero": ["cat", "dog", "rabbit", "fox", "bear", "mouse", "dragon", "robot", "star", "cloud"],
    "setting": ["forest", "moon", "village", "castle", "garden", "sea", "mountain", "school", "dreamland", "snowy field"],
    "problem": ["lost friend", "dark night", "broken toy", "missing key", "scary noise", "sad day", "big storm", "forgotten promise", "lonely creature", "hard choice"],
    "helper_or_item": ["magic leaf", "tiny lamp", "wise owl", "golden bell", "kind fairy", "talking stone", "little boat", "rainbow rope", "warm scarf", "silver star"],
    "lesson": ["kindness", "sharing", "bravery", "honesty", "patience", "friendship", "helping others", "not giving up", "saying sorry", "being careful"],
}
ASSISTANT_MAX_CHARS = 4000
PROMPT_ALLOWED = re.compile(r"^[\t\n\x20-\x7E]+$")
SEED_MAX = 2 ** 31 - 1

MODELS = [
    {"id": "backtalk-assistant-v2", "kind": KIND_ASSISTANT, "family": "ask", "version": "v2",
     "name": "Ask v2", "latest": True, "checkpoint": "checkpoints/backtalk-assistant-v2/model.pt",
     "default_max_new_tokens": 180},
    {"id": "backtalk-assistant-v1", "kind": KIND_ASSISTANT, "family": "ask", "version": "v1",
     "name": "Ask v1", "latest": False, "checkpoint": "checkpoints/backtalk-assistant/model.pt",
     "default_max_new_tokens": 180},
    {"id": "backtalk-storyteller-v1", "kind": KIND_STORYTELLER, "family": "stories", "version": "v1",
     "name": "Stories v1", "latest": True, "checkpoint": "checkpoints/backtalk-storyteller/model.pt",
     "default_max_new_tokens": 600},
]
MODELS_BY_ID = {m["id"]: m for m in MODELS}
ALIASES = {"backtalk-assistant": "backtalk-assistant-v2", "backtalk-storyteller": "backtalk-storyteller-v1"}


class ValidationError(ValueError):
    pass


class TokenizerCodec:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos = tokenizer.token_to_id(BOS_TOKEN)
        self.eos_id = tokenizer.token_to_id(EOS_TOKEN)
        self.stop_marker = None

    def prepare(self, prompt):
        reversed_prompt = reverse_text(prompt)
        ids = [self.bos] + self.tokenizer.encode(reversed_prompt).ids + [self.eos_id, self.bos]
        return prompt, reversed_prompt, ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)


class ByteCodec:
    eos_id = 256
    stop_marker = END_MARKER

    def prepare(self, story):
        input_text = "\n".join(f"{f}: {story[f]}" for f in INPUT_ORDER)
        model_input = f"<|input|>\n{reverse_text(input_text)}\n\n<|story|>\n"
        return input_text, model_input, list(model_input.encode("utf-8"))

    def decode(self, ids):
        return bytes(i for i in ids if i < self.eos_id).decode("utf-8", errors="replace")


class Runtime:
    def __init__(self, spec, model, config, codec, device):
        self.spec = spec
        self.model = model
        self.config = config
        self.codec = codec
        self.device = device

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.model.parameters())


class Engine:
    """Loads every model in MODELS and runs generation."""

    def __init__(self, tokenizer_path: str = DEFAULT_TOKENIZER, device: str = "auto"):
        self.device = pick_device(device)
        self.runtimes: Dict[str, Runtime] = {}
        tokenizer = None
        for spec in MODELS:
            path = os.path.join(ROOT, spec["checkpoint"])
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            config = GPTConfig(**ckpt["config"])
            config.gradient_checkpointing = False
            model = GPT(config).to(self.device)
            model.load_state_dict(ckpt["model"])
            model.eval()
            if spec["kind"] == KIND_ASSISTANT:
                if tokenizer is None:
                    tokenizer = load_tokenizer(tokenizer_path)
                codec = TokenizerCodec(tokenizer)
            else:
                codec = ByteCodec()
            self.runtimes[spec["id"]] = Runtime(spec, model, config, codec, self.device)
            del ckpt

    def advertised_models(self) -> List[dict]:
        """Concrete models (for the UI) plus alias entries (for gateway routing)."""
        out = []
        for spec in MODELS:
            rt = self.runtimes[spec["id"]]
            item = {k: spec[k] for k in ("id", "kind", "family", "version", "name", "latest", "default_max_new_tokens")}
            item["context_length"] = rt.config.block_size
            item["parameter_count"] = rt.parameter_count
            if spec["kind"] == KIND_STORYTELLER:
                item["story_options"] = STORY_OPTIONS
            out.append(item)
        for alias, target in ALIASES.items():
            spec = MODELS_BY_ID[target]
            out.append({"id": alias, "kind": spec["kind"], "family": spec["family"], "alias": True})
        return out

    def _resolve(self, model_id: str) -> Runtime:
        rid = ALIASES.get(model_id, model_id)
        rt = self.runtimes.get(rid)
        if rt is None:
            raise ValidationError(f"unknown model '{model_id}'")
        return rt

    def _clean_prompt(self, prompt: Optional[str]) -> str:
        text = (prompt or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            raise ValidationError("prompt is required")
        if len(text) > ASSISTANT_MAX_CHARS:
            raise ValidationError(f"prompt must be {ASSISTANT_MAX_CHARS} characters or fewer")
        if not PROMPT_ALLOWED.match(text):
            raise ValidationError("the assistant understands English only (Latin letters, digits, punctuation)")
        return text

    def _clean_story(self, story: Optional[dict]) -> dict:
        story = story or {}
        clean = {}
        for field in INPUT_ORDER:
            value = str(story.get(field, "")).strip()
            if value not in STORY_OPTIONS[field]:
                raise ValidationError(f"story.{field} must be one of {STORY_OPTIONS[field]}")
            clean[field] = value
        return clean

    def _events(self, model_id, prompt=None, story=None, params=None):
        """Yield generation events: a 'start', periodic 'chunk' (token count),
        then a final 'result'. Both generate() and generate_stream() use this."""
        rt = self._resolve(model_id)
        params = params or {}
        spec = rt.spec
        temperature = max(0.05, min(float(params.get("temperature", 0.7)), 2.0))
        top_k = max(1, min(int(params.get("top_k", 50)), rt.config.vocab_size))
        max_new = int(params.get("max_new_tokens") or spec["default_max_new_tokens"])
        max_new = max(8, min(max_new, 1000))
        seed = params.get("seed")
        seed = random.randint(0, SEED_MAX) if seed is None else int(seed) % (SEED_MAX + 1)
        set_seed(seed, rt.device)

        payload = self._clean_prompt(prompt) if rt.spec["kind"] == KIND_ASSISTANT else self._clean_story(story)
        readable_input, model_input, ids = rt.codec.prepare(payload)
        block = rt.config.block_size
        index = torch.tensor([ids[-block:]], dtype=torch.long, device=rt.device)
        generated: List[int] = []
        previous = ""
        yield {"event": "start", "model": spec["id"], "input": readable_input, "model_input": model_input}
        with torch.no_grad():
            for step in range(max_new):
                logits, _ = rt.model(index[:, -block:])
                logits = logits[:, -1, :] / temperature
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
                token = int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())
                if token == rt.codec.eos_id:
                    break
                generated.append(token)
                index = torch.cat([index, torch.tensor([[token]], dtype=torch.long, device=rt.device)], dim=1)
                decoded = rt.codec.decode(generated)
                raw = decoded.split(rt.codec.stop_marker, 1)[0] if rt.codec.stop_marker else decoded
                # Send only the real text added by this model step. A reset is
                # rare, but handles token decoders that revise their suffix.
                reset = not raw.startswith(previous)
                delta = raw if reset else raw[len(previous):]
                previous = raw
                yield {
                    "event": "chunk",
                    "delta": delta,
                    "reset": reset,
                    "tokens": len(generated),
                }
                if rt.codec.stop_marker and rt.codec.stop_marker in decoded:
                    break
        output = previous.rstrip("\n")
        yield {
            "event": "result", "model": spec["id"],
            "output": reverse_text(output), "raw_output": output,
            "input": readable_input, "model_input": model_input,
            "tokens": len(generated), "seed": seed,
        }

    def generate(self, model_id, prompt=None, story=None, params=None) -> dict:
        result = None
        for ev in self._events(model_id, prompt=prompt, story=story, params=params):
            if ev["event"] == "result":
                result = ev
        result.pop("event", None)
        return result

    def generate_stream(self, model_id, prompt=None, story=None, params=None):
        return self._events(model_id, prompt=prompt, story=story, params=params)
