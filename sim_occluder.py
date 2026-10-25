"""How many 'pixels' of a hidden scene can a room let you see around a corner?

Flatland simulation of the light-transport matrix A (hidden radiance -> visible
floor pixels), with and without an occluding edge, and its singular values.

  no edge   : every floor pixel sees every hidden point through a smooth
              Lambertian / distance falloff -> A is a smooth kernel
  door edge : a floor pixel at angle phi around the corner sees only hidden
              points at angle < phi (Bouman et al.'s corner camera geometry)
              -> each row of A contains a step

Prediction: smooth kernel -> singular values fall exponentially;
            step edges     -> they fall only like 1/k.
The number of singular values above the noise floor is how many independent
"pixels" of the hidden side the visible pixels can carry.
"""

import numpy as np


def transport(n_floor=200, n_hidden=120, occluder=True):
    phi = np.linspace(0.02, np.pi / 2 - 0.02, n_floor)       # floor pixel angles around the corner
    r_floor = 0.8
    floor = np.c_[r_floor * np.cos(phi), r_floor * np.sin(phi)]
    theta = np.linspace(0.02, np.pi / 2 - 0.02, n_hidden)    # hidden points, further away
    hidden = np.c_[3.0 * np.cos(theta), 3.0 * np.sin(theta)]
    d = hidden[None] - floor[:, None]
    dist = np.linalg.norm(d, axis=2)
    cos_floor = np.clip(d[..., 1] / dist + 0.5, 0, None)      # smooth incidence weighting
    A = cos_floor / dist**2
    if occluder:
        A = A * (theta[None, :] < phi[:, None])
    return A / np.linalg.norm(A, 2)


def count_above(s, floor):
    return int(np.sum(s / s[0] > floor))


def main():
    out = {}
    for name, occ in (("no edge", False), ("door edge", True)):
        s = np.linalg.svd(transport(occluder=occ), compute_uv=False)
        out[name] = s
        print(f"{name:10s}  usable pixels at noise 1e-2: {count_above(s, 1e-2):3d}   "
              f"at 1e-3: {count_above(s, 1e-3):3d}   sigma_10/sigma_1 = {s[9] / s[0]:.2e}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        k = np.arange(1, 61)
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, s in out.items():
            ax.semilogy(k, (s / s[0])[:60], "-o", ms=3, label=name)
        ax.semilogy(k, 1 / k, ":", color="#999", label="1/k")
        ax.axhline(1e-2, color="#c66", lw=0.8, ls="--")
        ax.text(42, 1.3e-2, "noise floor 1e-2", color="#c66", fontsize=8)
        ax.set_xlabel("pattern k")
        ax.set_ylabel("singular value (relative)")
        ax.set_title("What a room passes from the hidden side")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig("occluder_singular_values.png", dpi=130)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
