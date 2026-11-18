"""Python port of Wavefield_codes_v5.m: linearised (Born) inverse scattering.

A weakly scattering disc (k_obj = 1.05 k_b) is probed by 8 line sources at 9
frequencies (k_b = 1 ... 5) and measured by 150 receivers on a circle. Under the
Born approximation the scattered field is linear in the contrast chi:

    u_sc(rho_r) = k^2 h^2 * sum_n G(rho_r, rho_n) u_inc(rho_n) chi_n,
    G(r) = -(j/4) H0^(2)(k r)  (2-D Helmholtz Green's function)

and chi is recovered with Tikhonov regularisation.

Usage:  python wavefield_born.py [--save figures/born_reconstruction.png]
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hankel2


def green(k, r):
    return -0.25j * hankel2(0, k * r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=None)
    ap.add_argument("--snr-db", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # Geometry (same values as the MATLAB script)
    k_vec = np.arange(1, 5.01, 0.5)
    kb = 1.0
    lam0 = 2 * np.pi / kb
    h = lam0 / 20
    coords = np.arange(0, lam0 + h / 2, h)
    XD, YD = np.meshgrid(coords, coords)
    X, Y = XD.ravel(), YD.ravel()
    cx = cy = lam0 / 2

    k_obj = np.where(np.hypot(XD - cx, YD - cy) <= lam0 / 2, 1.05, 1.0)
    chi = (k_obj / kb) ** 2 - 1
    x_true = chi.ravel()

    th_s = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    src = np.c_[cx + 6 * lam0 * np.cos(th_s), cy + 6 * lam0 * np.sin(th_s)]
    th_r = np.linspace(0, 2 * np.pi, 150, endpoint=False)
    rec = np.c_[cx + 3 * lam0 * np.cos(th_r), cy + 3 * lam0 * np.sin(th_r)]

    # System matrix: one block of rows per (source, frequency)
    d_rn = np.hypot(rec[:, 0:1] - X[None, :], rec[:, 1:2] - Y[None, :])
    blocks = []
    for s in src:
        d_inc = np.hypot(X - s[0], Y - s[1])
        for k in k_vec:
            blocks.append(k ** 2 * green(k, d_rn) * green(k, d_inc)[None, :] * h ** 2)
    A = np.vstack(blocks)
    u = A @ x_true

    p_sig = np.linalg.norm(u) ** 2 / u.size
    p_noise = p_sig / 10 ** (args.snr_db / 10)
    u_noisy = u + np.sqrt(p_noise / 2) * (rng.standard_normal(u.size) + 1j * rng.standard_normal(u.size))

    s1 = np.linalg.svd(A, compute_uv=False)[0]
    AhA, I = A.conj().T @ A, np.eye(A.shape[1])

    def tikhonov(data, alpha):
        return np.linalg.solve(AhA + alpha * s1 ** 2 * I, A.conj().T @ data).real.reshape(chi.shape)

    rec_clean = tikhonov(u, 1e-5)
    rec_noisy = tikhonov(u_noisy, 1e-2)
    err = lambda r: np.linalg.norm(r - chi) / np.linalg.norm(chi)
    print(f"A: {A.shape[0]} x {A.shape[1]}")
    print(f"relative error, noiseless (alpha=1e-5): {err(rec_clean):.4f}")
    print(f"relative error, {args.snr_db:.0f} dB SNR (alpha=1e-2): {err(rec_noisy):.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    vmax = chi.max()
    for ax, img, title in zip(
        axes,
        [chi, rec_clean, rec_noisy],
        ["True contrast χ", f"Tikhonov, noiseless\nrel. error {err(rec_clean):.2f}",
         f"Tikhonov, {args.snr_db:.0f} dB SNR\nrel. error {err(rec_noisy):.2f}"],
    ):
        im = ax.imshow(img, extent=[coords[0], coords[-1], coords[-1], coords[0]], vmin=0, vmax=vmax * 1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x")
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel("y")
    fig.tight_layout()
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        fig.savefig(args.save, dpi=130)
    else:
        plt.show()


if __name__ == "__main__":
    main()
