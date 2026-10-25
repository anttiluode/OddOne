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

The picture is split down the middle like oddone.py: one half is your
camera, the other half is the AI version of the same frame, with a yellow
line between them.

Type a prompt in THIS console and press Enter: the filter rolls into it.

Sliders (top of the window)
  strength %   how much of you is replaced (15 = barely, 95 = almost all)
  roll frames  how many frames a new identity takes to roll in
  smooth %     blend with the previous AI frame (steadier, a bit ghostly)
  AI side      0 = AI on the left, 1 = AI on the right
  context      0 = the AI sees only its half (about 2x faster)
               1 = the AI sees the whole frame, so the seam lines up better

Keys
  n        roll to the next preset identity
  space    AI off (whole picture raw) / back on
  s        save a snapshot to aifilter_*.png
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

W, H = 640, 480          # both multiples of 8, as the model needs

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


def console_reader(q):
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            return
        if line.strip():
            q.put(line.strip())


def status_panel(lines, width):
    panel = np.full((22 * len(lines) + 12, width, 3), 24, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(panel, t, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (230, 230, 230), 1, cv2.LINE_AA)
    return panel


SLIDERS = (("strength %", 50, 95), ("roll frames", 40, 200), ("smooth %", 35, 90),
           ("AI side", 1, 1), ("context", 0, 1))


def slider(name):
    return cv2.getTrackbarPos(name, "aifilter")


# ------------------------------------------------------------------ app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--dshow", action="store_true", help="use DirectShow backend (Windows)")
    ap.add_argument("--model", default="stabilityai/sd-turbo")
    ap.add_argument("--token", default=None, help="Hugging Face token (else HF_TOKEN or `hf auth login`)")
    ap.add_argument("--prompt", default=PRESETS[0])
    ap.add_argument("--fake", action="store_true", help="no model, colour trick only (test the app)")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run until q)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if args.dshow else cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read from the camera. Try --camera 1 or --dshow.")

    filt = FakeFilter() if args.fake else TurboFilter(args.model, token=args.token)
    print(__doc__)

    cv2.namedWindow("aifilter", cv2.WINDOW_AUTOSIZE)
    for name, value, top in SLIDERS:
        cv2.createTrackbar(name, "aifilter", value, top, lambda v: None)

    current_prompt = args.prompt
    cur = filt.embed(current_prompt)
    src, dst = cur, cur
    roll_t = 1.0
    ai_on = True
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
    shown = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        while not q.empty():
            start_roll(q.get())

        strength = max(15, slider("strength %")) / 100.0
        roll_len = max(1, slider("roll frames"))
        smooth = slider("smooth %") / 100.0
        ai_right = slider("AI side") == 1
        whole = slider("context") == 1

        if roll_t < 1.0:
            roll_t = min(1.0, roll_t + 1.0 / roll_len)
            w = 0.5 - 0.5 * np.cos(np.pi * roll_t)      # ease in and out
            cur = lerp(src, dst, float(w))

        raw = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        half = W // 2
        sl = slice(half, W) if ai_right else slice(0, half)
        shown = raw.copy()
        if ai_on:
            src_img = raw if whole else raw[:, sl]
            rgb = cv2.cvtColor(np.ascontiguousarray(src_img), cv2.COLOR_BGR2RGB)
            res = cv2.cvtColor(filt(rgb, cur, strength), cv2.COLOR_RGB2BGR)
            if res.shape[:2] != src_img.shape[:2]:
                res = cv2.resize(res, (src_img.shape[1], src_img.shape[0]))
            ai_half = res[:, sl] if whole else res
            if smooth > 0 and prev is not None and prev.shape == ai_half.shape:
                ai_half = cv2.addWeighted(ai_half, 1 - smooth, prev, smooth, 0)
            prev = ai_half
            shown[:, sl] = ai_half
            cv2.line(shown, (half, 0), (half, H), (0, 255, 255), 1)

        n += 1
        if n % 10 == 0:
            now = time.time()
            fps, t_last = 10 / max(now - t_last, 1e-6), now
        state = ("AI off" if not ai_on else
                 "rolling %d%%" % int(100 * roll_t) if roll_t < 1 else "steady")
        info = [current_prompt[:80],
                f"{state}   AI on the {'right' if ai_right else 'left'}   "
                f"sees {'whole frame' if whole else 'its half'}   {fps:.1f} fps",
                "type a new identity in the console | n next preset | space AI off | s save | q quit"]
        cv2.imshow("aifilter", np.vstack([shown, status_panel(info, W)]))

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27) or (args.frames and n >= args.frames):
            break
        elif k == ord("n"):
            preset_i = (preset_i + 1) % len(PRESETS)
            start_roll(PRESETS[preset_i])
        elif k == ord(" "):
            ai_on, prev = not ai_on, None
        elif k == ord("s"):
            name = f"aifilter_{int(time.time())}.png"
            cv2.imwrite(name, shown)
            print("saved", name)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
