"""Independent numpy reference for the Jacobian used by the foam model.

Reads the JSON written by `examples/dump_reference.rs` and recomputes, with a *different*
algorithm from the crate, the height field, the displacement field and the Jacobian
eigenvalues:

  * displacement D is built as a spatial field from D~ = +i * k_hat * c * s * h~,
  * its derivatives are taken by spectral differentiation of that REAL spatial field
    (the crate instead folds i*k into one multiplier before a single inverse FFT),
  * lambda_min / det J then come from numpy's own eigen-solver.

Exits non-zero if anything disagrees beyond float32 tolerance.
"""

import json
import sys

import numpy as np


def main(path):
    d = json.load(open(path))
    n, size, c, s = d["n"], d["size"], d["choppiness"], d["scale"]
    h = np.array(d["spectrum"], dtype=np.float64)
    h = h[..., 0] + 1j * h[..., 1]  # h~[i, j], i = x bin, j = y bin

    freq = np.fft.fftfreq(n, d=1.0 / n)  # 0..n/2-1, -n/2..-1 : same layout as the crate
    kx = 2 * np.pi * freq[:, None] / size * np.ones((1, n))
    ky = 2 * np.pi * freq[None, :] / size * np.ones((n, 1))
    k = np.hypot(kx, ky)
    k_safe = np.where(k > 1e-6, k, 1.0)

    # Same normalisation as the crate: numpy's ifft2 already divides by n^2.
    height = np.real(np.fft.ifft2(h))
    dx = np.real(np.fft.ifft2(1j * (kx / k_safe) * c * h))
    dy = np.real(np.fft.ifft2(1j * (ky / k_safe) * c * h))
    # k = 0 contributes nothing (the crate's Phillips spectrum is 0 there)

    def ddx(f, kk):  # spectral derivative of a real field; Nyquist dropped
        F = np.fft.fft2(f)
        mask = np.ones((n, n))
        if n % 2 == 0:
            mask[n // 2, :] = 0
            mask[:, n // 2] = 0
        return np.real(np.fft.ifft2(1j * kk * F * mask))

    jxx = 1 + s * ddx(dx, kx)
    jxy = s * ddx(dx, ky)
    jyx = s * ddx(dy, kx)
    jyy = 1 + s * ddx(dy, ky)

    sym = 0.5 * (jxy + jyx)
    asym = np.abs(jxy - jyx).max()
    lam = np.empty((n, n))
    det = jxx * jyy - jxy * jyx
    for i in range(n):
        for j in range(n):
            lam[i, j] = np.linalg.eigvalsh(np.array([[jxx[i, j], sym[i, j]], [sym[i, j], jyy[i, j]]]))[0]

    rust_lam = np.array(d["lambda_min"])
    rust_det = np.array(d["det"])
    rust_h = np.array(d["height"])

    errs = {
        "height (x scale)": np.abs(s * height - rust_h).max(),
        "lambda_min": np.abs(lam - rust_lam).max(),
        "det J": np.abs(det - rust_det).max(),
    }
    signal = np.abs(rust_det - 1).max()
    print("max |J - 1| in the test field: %.3f (needs to be well above tolerance)" % signal)
    print("max |Dxy - Dyx| (symmetry): %.2e" % asym)
    ok = signal > 0.2 and asym < 1e-3
    for name, e in errs.items():
        print("%-18s max abs error %.2e" % (name, e))
        ok = ok and e < 1e-3
    if not ok:
        print("FAIL")
        return 1
    print("OK: independent numpy reference agrees with the crate")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
