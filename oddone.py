"""OddOne: can the visible half of a webcam learn to see the hidden half?

Split the webcam image in two. The VISIBLE half is the only input. The HIDDEN
half is the target, used only while learning ("ground zero, live"). Press H to
hide it and let the lens draw what it thinks is there.

Two lenses learn side by side, both scored honestly: every frame is predicted
BEFORE the lens learns from it (prequential scoring), so the score is always
on a frame the lens has never seen.

  linear   the best linear map from visible to hidden (Wiener / linear MMSE),
           learned from running covariances. Light transport is linear, so
           this is the physics-shaped lens: it reads shadows, glow and colour
           cast, whatever this room's geometry passes.
  cnn      a small convolutional net trained online (needs PyTorch). Stronger,
           but also better at GUESSING from habit (priors). See README for how
           to tell reading from guessing.

Score = R2 of the CHANGE in the hidden half, relative to its running average:
0 means no better than "it looks like it usually does"; 1 means perfect.

Keys
  space  learning on / off           h  hide the hidden half (show prediction)
  m      switch lens linear / cnn    c  count channels (how many independent
  e      try to lock auto-exposure      things the visible half tells)
  r      reset                       s / l  save / load the linear lens
  q, Esc quit
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import deque

import cv2
import numpy as np

LW, LH = 16, 24          # linear lens resolution per half (width, height)
NW, NH = 64, 96          # cnn resolution per half


# ------------------------------------------------------------------ halves


def split(frame, hidden="right"):
    h, w = frame.shape[:2]
    left, right = frame[:, : w // 2], frame[:, w // 2: 2 * (w // 2)]
    return (left, right) if hidden == "right" else (right, left)


def vec(img, size=(LW, LH)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0


def unvec(v, shape_hw, size=(LW, LH)):
    small = np.clip(v.reshape(size[1], size[0], 3) * 255, 0, 255).astype(np.uint8)
    return cv2.resize(small, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_CUBIC)


# ------------------------------------------------------------------ linear lens


class LinearLens:
    """x_hat = mean_x + W (y - mean_y), W = C_xy (C_yy + ridge)^-1, from forgetting running sums.

    The solve for W runs in a background thread so the camera loop never waits.
    """

    def __init__(self, din, dout, forget=0.9995, ridge=1e-2, solve_every=30, background=True):
        self.din, self.dout = din, dout
        self.forget, self.ridge, self.solve_every = forget, ridge, solve_every
        self.background = background
        self._thread = None
        self.reset()

    def wait(self):
        if self._thread is not None:
            self._thread.join()

    def reset(self):
        if getattr(self, "_thread", None) is not None:
            self._thread.join()
        self.w = 0.0
        self.sy = np.zeros(self.din, np.float32)
        self.sx = np.zeros(self.dout, np.float32)
        self.syy = np.zeros((self.din, self.din), np.float32)
        self.sxy = np.zeros((self.dout, self.din), np.float32)
        self.W = np.zeros((self.dout, self.din), np.float32)
        self.n = 0

    def means(self):
        w = max(self.w, 1e-9)
        return self.sy / w, self.sx / w

    def predict(self, y):
        my, mx = self.means()
        return mx + self.W @ (y - my)

    def learn(self, y, x):
        f = self.forget
        y = y.astype(np.float32, copy=False)
        x = x.astype(np.float32, copy=False)
        self.w = f * self.w + 1
        self.sy *= f
        self.sy += y
        self.sx *= f
        self.sx += x
        self.syy *= f
        self.syy += np.outer(y, y)
        self.sxy *= f
        self.sxy += np.outer(x, y)
        self.n += 1
        if self.n % self.solve_every == 0 and self.n > 20:
            if not self.background:
                self.solve()
            elif self._thread is None or not self._thread.is_alive():
                snap = (self.w, self.sy.copy(), self.sx.copy(), self.syy.copy(), self.sxy.copy())
                self._thread = threading.Thread(target=self.solve, args=(snap,), daemon=True)
                self._thread.start()

    def solve(self, snap=None):
        w, sy, sx, syy, sxy = snap if snap is not None else (self.w, self.sy, self.sx, self.syy, self.sxy)
        my, mx = sy / w, sx / w
        cyy = syy / w - np.outer(my, my)
        cxy = sxy / w - np.outer(mx, my)
        lam = self.ridge * float(np.trace(cyy)) / self.din + 1e-8
        self.W = np.linalg.solve(cyy + lam * np.eye(self.din, dtype=np.float32), cxy.T).T.astype(np.float32)

    def state(self):
        return dict(w=self.w, sy=self.sy, sx=self.sx, syy=self.syy, sxy=self.sxy, n=self.n)

    def load(self, d):
        self.w = float(d["w"])
        self.sy, self.sx = d["sy"].astype(np.float32), d["sx"].astype(np.float32)
        self.syy, self.sxy = d["syy"].astype(np.float32), d["sxy"].astype(np.float32)
        self.n = int(d["n"])
        self.solve()


# ------------------------------------------------------------------ cnn lens (optional)


class CnnLens:
    def __init__(self):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        c = 48
        self.net = nn.Sequential(
            nn.Conv2d(3, c, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(c, 2 * c, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(2 * c, 4 * c, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(4 * c, 4 * c, 3, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(4 * c, 2 * c, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(2 * c, c, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(c, 3, 4, 2, 1),
        ).to(self.dev)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.bg_x = None
        self.bg_y = None

    def _t(self, img):
        a = cv2.resize(img, (NW, NH), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        return self.torch.from_numpy(a).permute(2, 0, 1)[None].to(self.dev)

    def predict_img(self, vis, shape_hw):
        y = self._t(vis)
        if self.bg_x is None:
            self.bg_x, self.bg_y = self.torch.zeros_like(y), y.clone()
        with self.torch.no_grad():
            out = (self.bg_x + self.net(y - self.bg_y)).clamp(0, 1)
        a = (out[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return cv2.resize(a, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_CUBIC)

    def learn(self, vis, hid, steps=2):
        y, x = self._t(vis), self._t(hid)
        if self.bg_x is None:
            self.bg_x, self.bg_y = x.clone(), y.clone()
        self.bg_x.mul_(0.995).add_(0.005 * x)
        self.bg_y.mul_(0.995).add_(0.005 * y)
        for _ in range(steps):
            self.opt.zero_grad()
            loss = ((self.bg_x + self.net(y - self.bg_y) - x) ** 2).mean()
            loss.backward()
            self.opt.step()


# ------------------------------------------------------------------ channels (CCA)


def count_channels(Y, X, k=30, split_frac=0.75):
    """Held-out canonical correlations between visible and hidden halves.

    Each canonical pair is one independent 'thing' the visible half tells about
    the hidden half. Fitted on the first 75% of the buffer, scored on the rest.
    """
    n = len(Y)
    tr = int(n * split_frac)
    Y = Y - Y[:tr].mean(0)
    X = X - X[:tr].mean(0)
    def pca(A):
        _, _, Vt = np.linalg.svd(A[:tr], full_matrices=False)
        return A @ Vt[:k].T
    Yp, Xp = pca(Y), pca(X)
    def whiten(A):
        C = A[:tr].T @ A[:tr] / tr
        C = C + 1e-3 * np.trace(C) / C.shape[0] * np.eye(C.shape[0])   # ignore near-empty directions
        e, V = np.linalg.eigh(C)
        return V @ np.diag(e ** -0.5) @ V.T
    Wy, Wx = whiten(Yp), whiten(Xp)
    M = Wy @ (Yp[:tr].T @ Xp[:tr] / tr) @ Wx
    U, s, Vt = np.linalg.svd(M)
    a = Yp[tr:] @ Wy @ U
    b = Xp[tr:] @ Wx @ Vt.T
    corr = []
    for i in range(min(a.shape[1], b.shape[1])):
        ai, bi = a[:, i] - a[:, i].mean(), b[:, i] - b[:, i].mean()
        corr.append(float(ai @ bi / (np.linalg.norm(ai) * np.linalg.norm(bi) + 1e-12)))
    return np.array(corr)


# ------------------------------------------------------------------ scoring


class Score:
    """R2 of the change: 1 - sum|x - pred|^2 / sum|x - mean|^2, as slow running sums."""

    def __init__(self, rate=0.02):
        self.rate, self.num, self.den = rate, None, None

    def add(self, err2, var2):
        if self.num is None:
            self.num, self.den = err2, var2
        else:
            self.num += self.rate * (err2 - self.num)
            self.den += self.rate * (var2 - self.den)

    def value(self):
        return float("nan") if not self.den else 1 - self.num / self.den


# ------------------------------------------------------------------ app


def panel(real, pred, info_lines, channels, width):
    th, tw = 240, 160
    r = cv2.resize(real, (tw, th))
    p = cv2.resize(pred, (tw, th))
    e = cv2.resize(np.clip(np.abs(real.astype(int) - pred.astype(int)) * 3, 0, 255).astype(np.uint8), (tw, th))
    text = np.full((th, width - 3 * tw, 3), 24, np.uint8)
    for i, line in enumerate(info_lines):
        cv2.putText(text, line, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    if channels is not None:
        y0 = 22 + 20 * len(info_lines) + 6
        for i, c in enumerate(channels[:10]):
            L = int(max(c, 0) * (text.shape[1] - 50))
            col = (90, 200, 120) if c > 0.5 else (90, 150, 220) if c > 0.3 else (120, 120, 120)
            cv2.rectangle(text, (40, y0 + 12 * i), (40 + L, y0 + 12 * i + 9), col, -1)
            cv2.putText(text, f"{c:.2f}", (4, y0 + 12 * i + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)
    out = np.hstack([r, p, e, text])
    for i, label in enumerate(["hidden (real)", "lens", "|error| x3"]):
        cv2.putText(out, label, (6 + i * tw, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)
    return out


def try_lock_exposure(cap):
    before = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    for v in (0.25, 1):            # DirectShow manual, V4L2 manual
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, v)
    return before, cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--dshow", action="store_true", help="use DirectShow backend (Windows)")
    ap.add_argument("--hidden", choices=["right", "left"], default="right")
    ap.add_argument("--no-cnn", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if args.dshow else cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read from the camera. Try --camera 1 or --dshow.")

    din = dout = LW * LH * 3
    linear = LinearLens(din, dout)
    cnn = None
    if not args.no_cnn:
        try:
            cnn = CnnLens()
        except Exception as e:  # torch missing or broken
            print("CNN lens unavailable:", e)
    s_lin, s_cnn = Score(), Score()
    buf = deque(maxlen=3000)
    learning, hide, mode = True, False, "linear"
    channels, n_frames, t_last, fps = None, 0, time.time(), 0.0
    print(__doc__)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis, hid = split(frame, args.hidden)
        y, x = vec(vis), vec(hid)

        # predict first (honest), then score, then learn
        pred_lin = linear.predict(y)
        _, mx = linear.means()
        if linear.n > 50:
            s_lin.add(float(np.sum((x - pred_lin) ** 2)), float(np.sum((x - mx) ** 2)))
        pred_lin_img = unvec(pred_lin, hid.shape[:2])
        pred_cnn_img = None
        if cnn is not None and cnn.bg_x is not None:
            pred_cnn_img = cnn.predict_img(vis, hid.shape[:2])
            if linear.n > 50:
                pc = vec(pred_cnn_img)
                s_cnn.add(float(np.sum((x - pc) ** 2)), float(np.sum((x - mx) ** 2)))

        if learning:
            linear.learn(y, x)
            if cnn is not None:
                cnn.learn(vis, hid)
            buf.append((y, x))

        pred_img = pred_cnn_img if (mode == "cnn" and pred_cnn_img is not None) else pred_lin_img
        shown_hidden = pred_img if hide else hid
        shown = np.hstack([vis, shown_hidden]) if args.hidden == "right" else np.hstack([shown_hidden, vis])
        mid = vis.shape[1]
        cv2.line(shown, (mid, 0), (mid, shown.shape[0]), (0, 255, 255), 1)
        if hide:
            hx = mid + 8 if args.hidden == "right" else 8
            cv2.putText(shown, f"PREDICTED ({mode})", (hx, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        n_frames += 1
        if n_frames % 10 == 0:
            now = time.time()
            fps = 10 / max(now - t_last, 1e-6)
            t_last = now
        info = [
            f"learning: {'ON' if learning else 'off'}   lens: {mode}   fps {fps:.0f}",
            f"frames learned: {linear.n}",
            f"R2 of change, linear: {s_lin.value():.3f}",
            f"R2 of change, cnn:    {s_cnn.value():.3f}" if cnn else "cnn: unavailable",
            "0 = no better than the usual look",
            "channels (press c): held-out corr." if channels is None else
            f"channels >0.5: {int((channels > 0.5).sum())}   >0.3: {int((channels > 0.3).sum())}",
        ]
        bottom = panel(hid, pred_img, info, channels, shown.shape[1])
        cv2.imshow("OddOne", np.vstack([shown, bottom]))

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord(" "):
            learning = not learning
        elif k == ord("h"):
            hide = not hide
        elif k == ord("m") and cnn is not None:
            mode = "cnn" if mode == "linear" else "linear"
        elif k == ord("c"):
            if len(buf) < 300:
                print(f"Need at least 300 learned frames for channels (have {len(buf)}).")
            else:
                Y = np.array([b[0] for b in buf])
                X = np.array([b[1] for b in buf])
                channels = count_channels(Y, X)
                print("held-out canonical correlations:", np.round(channels[:12], 3))
        elif k == ord("e"):
            print("auto-exposure before/after:", try_lock_exposure(cap))
        elif k == ord("r"):
            linear.reset()
            s_lin, s_cnn = Score(), Score()
            buf.clear()
            channels = None
            if cnn is not None:
                cnn = CnnLens()
        elif k == ord("s"):
            np.savez("oddone_linear.npz", **linear.state())
            print("saved oddone_linear.npz")
        elif k == ord("l"):
            try:
                linear.load(np.load("oddone_linear.npz"))
                print("loaded oddone_linear.npz")
            except FileNotFoundError:
                print("no oddone_linear.npz yet")

    linear.wait()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
