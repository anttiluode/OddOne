"""aifilter.py: roll into another identity, live.

The fun side dish of OddOne, not the experiment. Your webcam frame goes
through SD-Turbo image-to-image with a text prompt. Your pose, light and
movement are kept; the identity painted over them is whatever you type.

When you type a new prompt, the filter does not jump. It ROLLS: the text
embedding slides from the old identity to the new one over a number of
frames, so you watch yourself melt from one person (or creature) into the
next.

Run
  pip install torch diffusers transformers accelerate   (plus opencv-python, numpy)
  python aifilter.py                  # Windows: add --dshow if the camera is slow to open
  python aifilter.py --fake           # no model: tests the window and keys with a colour trick

Type a prompt in THIS console and press Enter: the filter rolls into it.

Keys (in the video window)
  n        roll to the next preset identity
  [ / ]    less / more transformation (strength)
  , / .    slower / faster roll
  t        temporal smoothing on / off (steadier, a bit ghostly)
  space    show the raw camera (filter off) / back on
  s        save a snapshot (raw | filtered) to aifilter_*.png
  q, Esc   quit
"""

from __future__ import annotations

import argparse
import os
import queue
import threading
import time

import cv2
import numpy as np

PRESETS = [
    "portrait of an old sea captain, weathered face, grey beard, photo",
    "portrait of a brass robot with glowing eyes, steampunk, photo",
    "portrait of a forest elf, pointed ears, moss and leaves, fantasy photo",
    "a marble statue of a person, classical sculpture, museum light",
    "portrait of a person made of ice crystals, frozen, cold blue light",
    "oil painting portrait in the style of a 17th century dutch master",
    "portrait of an alien with green skin and large dark eyes, photo",
    "cartoon character portrait, pixar style, 3d render",
]

SIZE = 512

HF_HELP = """
Could not load {model} from Hugging Face:
  {err}

SD-Turbo is a gated model: you need a free Hugging Face account, to accept its
licence once, and a token on this computer.

  1. Log in at https://huggingface.co and open https://huggingface.co/{model}
     Click "Agree and access repository".
  2. Make a token: https://huggingface.co/settings/tokens
     -> "Create new token", type "Read". Copy it (starts with hf_).
  3. Give it to this computer, ONE of these:
       hf auth login                       (paste the token; remembered for good)
       set HF_TOKEN=hf_xxx                 (Windows cmd, this window only)
       python aifilter.py --token hf_xxx   (this run only)

Never commit the token to git or paste it into a chat.
Want to test the window without any model?  python aifilter.py --fake
"""


# ------------------------------------------------------------------ the filter


class TurboFilter:
    """SD-Turbo img2img with prompt-embedding interpolation (the roll)."""

    def __init__(self, model="stabilityai/sd-turbo", steps=2, seed=1234, token=None):
        import torch
        from diffusers import AutoPipelineForImage2Image

        self.torch = torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if dev == "cuda" else torch.float32
        if dev == "cpu":
            print("No CUDA GPU found: this will be very slow on CPU.")
        print(f"Loading {model} on {dev} (first run downloads ~2.5 GB)...")
        token = token or os.environ.get("HF_TOKEN") or None
        try:
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                model, torch_dtype=dtype, variant="fp16" if dev == "cuda" else None, token=token)
        except OSError as e:
            raise SystemExit(HF_HELP.format(model=model, err=str(e).splitlines()[0])) from None
        self.pipe = self.pipe.to(dev)
        self.pipe.set_progress_bar_config(disable=True)
        self.dev, self.steps, self.seed = dev, steps, seed

    def embed(self, prompt):
        with self.torch.no_grad():
            pe, _ = self.pipe.encode_prompt(prompt, self.dev, 1, False)
        return pe

    def __call__(self, rgb, embeds, strength):
        from PIL import Image

        img = Image.fromarray(rgb)
        # SD-Turbo needs steps * strength >= 1; raise steps as strength drops
        steps = max(self.steps, int(np.ceil(1.0 / max(strength, 0.05))))
        # the same noise every frame keeps the identity steady while you move
        gen = self.torch.Generator(self.dev).manual_seed(self.seed)
        with self.torch.no_grad():
            out = self.pipe(prompt_embeds=embeds, image=img, strength=strength,
                            num_inference_steps=steps, guidance_scale=0.0,
                            generator=gen).images[0]
        return np.array(out)


class FakeFilter:
    """No model: rotates colour with the 'prompt' so the window and keys can be tested."""

    def embed(self, prompt):
        h = (sum(map(ord, prompt)) % 180)
        return np.array([h], np.float32)

    def __call__(self, rgb, embeds, strength):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + float(embeds[0]) * strength) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def lerp(a, b, w):
    return a * (1 - w) + b * w


# ------------------------------------------------------------------ helpers


def square(frame):
    h, w = frame.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return cv2.resize(frame[y0:y0 + s, x0:x0 + s], (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def console_reader(q):
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            return
        if line.strip():
            q.put(line.strip())


def label(img, lines):
    out = img.copy()
    for i, t in enumerate(lines):
        y = 22 + 22 * i
        cv2.putText(out, t, (9, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
    return out


# ------------------------------------------------------------------ app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--dshow", action="store_true", help="use DirectShow backend (Windows)")
    ap.add_argument("--model", default="stabilityai/sd-turbo")
    ap.add_argument("--token", default=None, help="Hugging Face token (else HF_TOKEN or `hf auth login`)")
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--roll", type=int, default=40, help="frames to roll from one identity to the next")
    ap.add_argument("--prompt", default=PRESETS[0])
    ap.add_argument("--fake", action="store_true", help="no model, colour trick only (test the app)")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run until q)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if args.dshow else cv2.VideoCapture(args.camera)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read from the camera. Try --camera 1 or --dshow.")

    filt = FakeFilter() if args.fake else TurboFilter(args.model, token=args.token)
    print(__doc__)

    current_prompt = args.prompt
    cur = filt.embed(current_prompt)
    src, dst = cur, cur
    roll_t, roll_len = 1.0, max(args.roll, 1)
    strength, smooth, show_raw = args.strength, True, False
    preset_i = PRESETS.index(args.prompt) if args.prompt in PRESETS else -1
    prev = None
    q = queue.Queue()
    threading.Thread(target=console_reader, args=(q,), daemon=True).start()
    print(f"\nNow: {current_prompt}\nType a new identity and press Enter:")

    def start_roll(prompt):
        nonlocal src, dst, roll_t, current_prompt
        src = cur                      # roll from wherever we are right now
        dst = filt.embed(prompt)
        roll_t = 0.0
        current_prompt = prompt
        print(f"rolling into: {prompt}")

    n, t_last, fps = 0, time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        while not q.empty():
            start_roll(q.get())

        if roll_t < 1.0:
            roll_t = min(1.0, roll_t + 1.0 / roll_len)
            w = 0.5 - 0.5 * np.cos(np.pi * roll_t)      # ease in and out
            cur = lerp(src, dst, float(w))

        raw = square(frame)
        if show_raw:
            out = raw
        else:
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            res = cv2.cvtColor(filt(rgb, cur, strength), cv2.COLOR_RGB2BGR)
            if res.shape[:2] != raw.shape[:2]:
                res = cv2.resize(res, (SIZE, SIZE))
            if smooth and prev is not None:
                res = cv2.addWeighted(res, 0.65, prev, 0.35, 0)
            prev = res
            out = res

        n += 1
        if n % 10 == 0:
            now = time.time()
            fps, t_last = 10 / max(now - t_last, 1e-6), now
        state = "RAW" if show_raw else ("rolling %d%%" % int(100 * roll_t) if roll_t < 1 else "steady")
        shown = label(out, [current_prompt[:58],
                            f"{state}   strength {strength:.2f}   roll {roll_len}f   "
                            f"smooth {'on' if smooth else 'off'}   {fps:.1f} fps"])
        cv2.imshow("aifilter", np.hstack([raw, shown]))

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27) or (args.frames and n >= args.frames):
            break
        elif k == ord("n"):
            preset_i = (preset_i + 1) % len(PRESETS)
            start_roll(PRESETS[preset_i])
        elif k == ord("["):
            strength = max(0.15, round(strength - 0.05, 2))
        elif k == ord("]"):
            strength = min(0.95, round(strength + 0.05, 2))
        elif k == ord(","):
            roll_len = int(roll_len * 1.5) + 1
        elif k == ord("."):
            roll_len = max(1, int(roll_len / 1.5))
        elif k == ord("t"):
            smooth, prev = not smooth, None
        elif k == ord(" "):
            show_raw = not show_raw
        elif k == ord("s"):
            name = f"aifilter_{int(time.time())}.png"
            cv2.imwrite(name, np.hstack([raw, out]))
            print("saved", name)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
