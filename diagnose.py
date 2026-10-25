"""diagnose.py: what is the OddOne lens actually doing?

Runs the real lens code (LinearLens, CnnLens, count_channels from oddone.py)
and writes a JSON report with an automatic verdict for each run.

  python diagnose.py                 # synthetic battery: 6 scenes with known answers
  python diagnose.py --live 120      # your camera: learn for 3/4 of the time, then freeze and test
  python diagnose.py --no-cnn        # skip the CNN lens

Synthetic scenes (a bright object flies around the HIDDEN right half):
  independent  the visible half has its own unrelated motion: nothing leaks (control)
  glow         the object's light falls smoothly on the visible wall (bare wall)
  shadow       same light, plus a pole in the visible half casting a sharp shadow
  crosser      a big object flies across the midline (continuation, not reading)
  exposure     a lamp in the hidden half switches brighter and dimmer; no light
               reaches the visible half, but the camera's auto-exposure dims the
               whole frame in response
  novel        'shadow' scene, but tested with an object of a NEW colour and size

Every run reports, on frozen test frames the lens never learned from:
  r2_linear          the real linear lens
  r2_brightness_only a lens that only sees the visible half's average colour
                     (if this is as good as r2_linear, it's global light or exposure)
  r2_shuffled        the linear lens trained on time-shuffled pairs (should be ~0)
  r2_cnn             the CNN lens, if PyTorch is available
  channels           held-out canonical correlations between the halves
  midline_motion     motion right at the line on the visible side, relative to motion
                     in the hidden half (above 0.1: things are crossing; continuation)
  visible_change_global_gain
                     share of the visible half's change explained by one overall
                     brightness factor (near 1: exposure or uniform light)
  test_novelty       share of the visible change during testing that lies outside
                     anything seen in training (high: the lens is extrapolating)
  exposure_corr      correlation of the two halves' brightness changes
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import cv2
import numpy as np

from oddone import LH, LW, LinearLens, coarse, count_channels, split, vec

H, W = 480, 640
D = LW * LH * 3


# ------------------------------------------------------------------ analysis


def r2_change(pred, X, mean_x):
    return float(1 - np.sum((X - pred) ** 2) / max(np.sum((X - mean_x) ** 2), 1e-12))


def ridge_fit(A, B, ridge=1e-2):
    ma, mb = A.mean(0), B.mean(0)
    Ac, Bc = A - ma, B - mb
    C = Ac.T @ Ac / len(A)
    lam = ridge * np.trace(C) / C.shape[0] + 1e-8
    Wt = np.linalg.solve(C + lam * np.eye(C.shape[0]), Ac.T @ Bc / len(A))
    return lambda Z: mb + (Z - ma) @ Wt


def analyse(Y, X, n_train, frames_stats, cnn_r2=None):
    """Y: visible vectors, X: hidden vectors (low-res), first n_train used for learning."""
    tr, te = slice(0, n_train), slice(n_train, None)
    mean_x = X[tr].mean(0)

    lens = LinearLens(D, D, background=False)       # the app's lens, app's settings
    for y, x in zip(Y[tr], X[tr]):
        lens.learn(y, x)
    lens.solve()
    pred = np.array([lens.predict(y) for y in Y[te]])
    r2_lin = r2_change(pred, X[te], mean_x)
    by_scale = {f"1/{f}": round(r2_change(coarse(pred, f), coarse(X[te], f), coarse(mean_x, f)), 3)
                for f in (1, 2, 4, 8)}

    bright = lambda Z: Z.reshape(len(Z), -1, 3).mean(1)          # average colour of visible half
    f = ridge_fit(bright(Y[tr]), X[tr])
    r2_bright = r2_change(f(bright(Y[te])), X[te], mean_x)

    rng = np.random.default_rng(0)
    perm = rng.permutation(n_train)
    sh = LinearLens(D, D, background=False)
    for y, x in zip(Y[tr][perm], X[tr]):
        sh.learn(y, x)
    sh.solve()
    r2_shuf = r2_change(np.array([sh.predict(y) for y in Y[te]]), X[te], mean_x)

    Yc = Y[tr] - Y[tr].mean(0)
    _, sv, Vt = np.linalg.svd(Yc, full_matrices=False)
    k = int(np.searchsorted(np.cumsum(sv ** 2) / np.sum(sv ** 2), 0.999)) + 1
    dte = Y[te] - Y[tr].mean(0)
    inside = dte @ Vt[:k].T
    novelty = float(1 - np.sum(inside ** 2) / max(np.sum(dte ** 2), 1e-12))
    ch = count_channels(Y, X)
    out = {
        "r2_linear": round(r2_lin, 3),
        "r2_linear_by_scale": by_scale,
        "ridge_chosen": lens.ridge,
        "r2_brightness_only": round(r2_bright, 3),
        "r2_shuffled": round(r2_shuf, 3),
        "r2_cnn": None if cnn_r2 is None else round(cnn_r2, 3),
        "test_novelty": round(novelty, 3),
        "channels_top10": [round(float(c), 3) for c in ch[:10]],
        "channels_over_0.5": int((ch > 0.5).sum()),
        "channels_over_0.3": int((ch > 0.3).sum()),
        **frames_stats,
    }
    out["verdict"] = verdict(out)
    return out


def verdict(r):
    scales = r.get("r2_linear_by_scale", {"1/1": r["r2_linear"]})
    best_scale = max(scales, key=scales.get)
    lin, br = scales[best_scale], r["r2_brightness_only"]
    if r["r2_shuffled"] > 0.1 and r["r2_shuffled"] > 0.5 * lin:
        return "WARNING: shuffled control scores too high; the hidden half barely changes or something is wrong"
    if lin < 0.05 and r.get("test_novelty", 0) > 0.1:
        return (f"unfamiliar input: {100 * r['test_novelty']:.0f}% of the visible change during testing lies "
                f"outside anything seen in training, and the lens extrapolates badly (R2 {lin:.2f})")
    if lin < 0.05:
        return "no usable leak: the visible half tells (almost) nothing about the hidden half's changes"
    if r.get("midline_motion", 0) > 0.1:
        return (f"continuation: much of the motion is at the midline, so the lens can continue objects "
                f"across it (R2 {lin:.2f}); not evidence of reading light")
    if br >= 0.8 * lin:
        g = r.get("visible_change_global_gain", 0)
        tag = ("the whole visible half scales together: auto-exposure or uniform room light"
               if g > 0.8 else "overall light level")
        return (f"global brightness: an average-colour lens does as well ({br:.2f} vs {lin:.2f}); "
                f"{tag}, not spatial structure")
    return (f"reading spatial light structure: R2 {lin:.2f} at {best_scale} resolution vs {br:.2f} "
            f"from brightness alone, {r['channels_over_0.5']} strong channels")


def frame_stats(frames_vis_mean, frames_hid_mean, motion_mid, motion_hid):
    dv, dh = np.diff(frames_vis_mean), np.diff(frames_hid_mean)
    ec = float(np.corrcoef(dv, dh)[0, 1]) if dv.std() > 1e-9 and dh.std() > 1e-9 else 0.0
    return {"exposure_corr": round(ec, 3),
            "midline_motion": round(float(motion_mid / motion_hid) if motion_hid > 0 else 0.0, 3)}


class Recorder:
    """Collects low-res vectors and the frame statistics the verdict needs."""

    def __init__(self):
        self.Y, self.X, self.vm, self.hm = [], [], [], []
        self.mid, self.hid = 0.0, 0.0
        self.prev = None
        self.gain_num, self.gain_den = 0.0, 0.0
        self.vis_ref = None

    def add(self, frame):
        vis, hid = split(frame)
        self.Y.append(vec(vis))
        self.X.append(vec(hid))
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.vm.append(float(g[:, : W // 2].mean()))
        self.hm.append(float(g[:, W // 2:].mean()))
        if self.prev is not None:
            moving = np.abs(g - self.prev) > 18          # above sensor noise
            self.mid += float(moving[:, W // 2 - 24: W // 2].sum())   # visible side, right at the line
            self.hid += float(moving[:, W // 2:].sum())
        self.prev = g
        # how much of the visible half's change is a single global gain (exposure / uniform light)?
        v = g[:, : W // 2]
        if self.vis_ref is None:
            self.vis_ref = v.copy()
        else:
            dv = v - self.vis_ref
            ref = self.vis_ref - self.vis_ref.mean()
            gscale = float((dv * self.vis_ref).sum() / max((self.vis_ref ** 2).sum(), 1e-9))
            resid = dv - gscale * self.vis_ref
            self.gain_num += float((dv ** 2).sum() - (resid ** 2).sum())
            self.gain_den += float(((dv - dv.mean()) ** 2).sum()) + 1e-9
            self.vis_ref = 0.98 * self.vis_ref + 0.02 * v

    def arrays(self):
        return np.array(self.Y), np.array(self.X)

    def stats(self):
        st = frame_stats(np.array(self.vm), np.array(self.hm), self.mid, self.hid)
        st["visible_change_global_gain"] = round(max(0.0, min(1.0, self.gain_num / self.gain_den)), 3)
        return st


# ------------------------------------------------------------------ optional CNN


def cnn_score(frames, n_train):
    try:
        from oddone import CnnLens
        cnn = CnnLens()
    except Exception as e:
        print("  (CNN skipped:", e, ")")
        return None
    for f in frames[:n_train]:
        vis, hid = split(f)
        cnn.learn(vis, hid)
    X = np.array([vec(split(f)[1]) for f in frames[n_train:]])
    P = np.array([vec(cnn.predict_img(split(f)[0], (H, W // 2))) for f in frames[n_train:]])
    mean_x = np.array([vec(split(f)[1]) for f in frames[:n_train]]).mean(0)
    return r2_change(P, X, mean_x)


# ------------------------------------------------------------------ synthetic scenes


def room(rng):
    """A static, softly textured room."""
    base = rng.uniform(40, 110, (H // 16, W // 16, 3)).astype(np.float32)
    return cv2.resize(base, (W, H), interpolation=cv2.INTER_CUBIC)


def path(t, rng_phase, x0, x1, y0, y1, speed=1.0):
    a, b, c = rng_phase
    x = x0 + (x1 - x0) * 0.5 * (1 + np.sin(speed * 0.031 * t + a) * np.cos(speed * 0.013 * t + b))
    y = y0 + (y1 - y0) * 0.5 * (1 + np.sin(speed * 0.023 * t + c))
    return x, y


YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
POLE_X, POLE_TOP = 230.0, 170.0


def render(scene, t, bg, rng, ph, color=(60, 160, 255), radius=34):
    img = bg.copy()
    bx, by = path(t, ph, W // 2 + 50, W - 50, 60, H - 60)
    color = np.array(color, np.float32)
    if scene == "crosser":
        cx, cy = path(t, ph, W // 2 - 60, W // 2 + 60, 120, H - 120)
        cv2.circle(img, (int(cx), int(cy)), 95, tuple(float(c) for c in color), -1)
        return finish(img, rng)
    if scene == "exposure":
        # a lamp in the hidden half flickers; the camera's metering dims the WHOLE frame in response
        lamp = 0.5 + 0.5 * np.sin(t / 30.0 + ph[0])
        cv2.circle(img, (int(W * 0.75), int(H * 0.4)), 70, tuple(float(c) * (0.4 + 1.6 * lamp) for c in (200, 220, 255)), -1)
        img = img / (1.0 + 0.6 * lamp)
        return finish(img, rng)
    cv2.circle(img, (int(bx), int(by)), radius, tuple(float(c) for c in color), -1)
    if scene == "independent":
        ox, oy = path(t * 1.37 + 500, ph[::-1], 50, W // 2 - 90, 60, H - 60)
        cv2.circle(img, (int(ox), int(oy)), 30, (200, 120, 90), -1)
    if scene in ("glow", "shadow", "novel"):
        d2 = (XX - bx) ** 2 + (YY - by) ** 2
        light = 9000.0 * (radius / 34) ** 2 / (d2 + 9000.0)
        if scene in ("shadow", "novel"):
            # a pole in the visible half: points beyond it, seen from the light, are in shadow
            with np.errstate(divide="ignore", invalid="ignore"):
                tcross = (POLE_X - bx) / (XX - bx)
                ycross = by + tcross * (YY - by)
            shadow = (XX < POLE_X) & (ycross >= POLE_TOP) & (ycross <= H)
            light = light * (~shadow)
            cv2.line(img, (int(POLE_X), int(POLE_TOP)), (int(POLE_X), H), (30, 30, 30), 6)
        vis = XX < W // 2
        img = img + (light * vis)[..., None] * color[None, None, :] * 0.55
    return finish(img, rng)


def finish(img, rng):
    img = img + rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def run_scene(scene, n_train, n_test, use_cnn, seed=0):
    rng = np.random.default_rng(seed)
    bg = room(rng)
    ph = rng.uniform(0, 2 * np.pi, 3)
    rec = Recorder()
    frames = []
    for t in range(n_train + n_test):
        if scene == "novel" and t >= n_train:
            f = render("novel", t, bg, rng, ph, color=(255, 90, 40), radius=26)     # new colour and size
        else:
            f = render("shadow" if scene == "novel" else scene, t, bg, rng, ph)
        rec.add(f)
        if use_cnn:
            frames.append(f)
    Y, X = rec.arrays()
    cnn_r2 = cnn_score(frames, n_train) if use_cnn else None
    return analyse(Y, X, n_train, rec.stats(), cnn_r2)


EXPECTED = {
    "independent": "no usable leak",
    "glow": "some reading, few channels (a smooth wall passes little)",
    "shadow": "more reading and more channels than glow (the edge passes structure)",
    "crosser": "flagged as continuation",
    "exposure": "flagged as global brightness / auto-exposure",
    "novel": "a linear lens trained on one colour of light should fail on a new colour, and be flagged as unfamiliar",
}


def synthetic(use_cnn, n_train=1200, n_test=400):
    report = {}
    for scene in EXPECTED:
        t0 = time.time()
        r = run_scene(scene, n_train, n_test, use_cnn)
        r["expected"] = EXPECTED[scene]
        report[scene] = r
        print(f"{scene:12s} R2 lin {r['r2_linear']:6.3f}  bright {r['r2_brightness_only']:6.3f}  "
              f"shuf {r['r2_shuffled']:6.3f}  ch>0.5 {r['channels_over_0.5']:2d}  "
              f"({time.time() - t0:.0f}s)\n             -> {r['verdict']}")
    return report


# ------------------------------------------------------------------ live


def live(seconds, camera, dshow, use_cnn):
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW) if dshow else cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    rec = Recorder()
    frames = []
    t_learn = seconds * 0.75
    print(f"LIVE: learning for {t_learn:.0f}s, then testing for {seconds - t_learn:.0f}s.")
    print("Keep everything that moves in the RIGHT half. Nothing should cross the yellow line.")
    t0 = time.time()
    n_train = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (W, H))
        el = time.time() - t0
        if el > seconds:
            break
        if n_train is None and el > t_learn:
            n_train = len(rec.Y)
        rec.add(f)
        if use_cnn:
            frames.append(f)
        show = f.copy()
        cv2.line(show, (W // 2, 0), (W // 2, H), (0, 255, 255), 1)
        phase = "LEARN" if n_train is None else "TEST (frozen)"
        cv2.putText(show, f"{phase}  {seconds - el:4.0f}s left", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)
        cv2.imshow("OddOne diagnose", show)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
    Y, X = rec.arrays()
    if n_train is None or len(Y) - n_train < 50:
        raise SystemExit("Not enough frames recorded; run longer.")
    cnn_r2 = cnn_score(frames, n_train) if use_cnn else None
    r = analyse(Y, X, n_train, rec.stats(), cnn_r2)
    r["frames"] = {"learn": int(n_train), "test": int(len(Y) - n_train),
                   "fps": round(len(Y) / max(time.time() - t0, 1e-6), 1)}
    print(json.dumps(r, indent=1))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", type=float, default=None, help="seconds of live camera diagnosis")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--dshow", action="store_true")
    ap.add_argument("--no-cnn", action="store_true")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    use_cnn = not args.no_cnn
    if args.live:
        report = {"mode": "live", "result": live(args.live, args.camera, args.dshow, use_cnn)}
        name = "diagnose_live.json"
    else:
        report = {"mode": "synthetic", "scenes": synthetic(use_cnn)}
        name = "diagnose_synthetic.json"
    with open(name, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"\nwrote {name}  (paste it to Claude)")


if __name__ == "__main__":
    main()
