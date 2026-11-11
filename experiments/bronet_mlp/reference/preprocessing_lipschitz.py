"""Lipschitz constants (L2 operator norms) of the trichoscopy preprocessing chain.

Chain of the dataset's preprocessing:
  raw 1920x1080 -> crop 420 px each side -> colour normalisation vs reference
  (x_c += int(0.8*(m_ref_c - m_c)) + (g_ref - g), clip) -> JPEG -> PIL bilinear
  resize to 600x600 -> JPEG q75 -> torchvision Resize(224) (= PIL bilinear) ->
  ToTensor (/255) -> Normalize(ImageNet mean/std)  [paper models only; BRONet
  takes [0,1] and subtracts 0.5 itself].

Linear steps are measured exactly; JPEG, integer rounding and clipping are
discussed in the write-up (JPEG and rounding are not Lipschitz at all).
Run:  python pretrained/preprocessing_lipschitz.py
"""
import numpy as np
from PIL import Image


def resize_1d(n, m):
    """The 1-D operator of PIL's bilinear resize n -> m (float 'F' mode is exact,
    no rounding). PIL resizes separably, so the 2-D operator is A kron A and its
    norm is sigma_max(A)**2."""
    A = np.zeros((m, n))
    for i in range(n):
        e = np.zeros((1, n), np.float32)
        e[0, i] = 1
        A[:, i] = np.asarray(Image.fromarray(e, mode="F").resize((m, 1), Image.BILINEAR))[0]
    return A


def resize_norm(n, m):
    return np.linalg.svd(resize_1d(n, m), compute_uv=False)[0] ** 2


def colour_norm_singular_values(h=8, w=8):
    """Linear part of the colour normalisation without rounding/clipping:
    x_c -> x_c - 0.8 * mean_c(x) - mean_all(x). Size-independent spectrum."""
    n = h * w * 3
    M = np.zeros((n, n))
    for k in range(n):
        x = np.zeros((h, w, 3))
        x.flat[k] = 1
        M[:, k] = (x - 0.8 * x.mean((0, 1)) - x.mean()).ravel()
    return np.unique(np.round(np.linalg.svd(M, compute_uv=False), 6))


if __name__ == "__main__":
    r600 = resize_norm(1080, 600)
    r224 = resize_norm(600, 224)
    inv_std = 1 / min(0.229, 0.224, 0.225)
    print(f"crop                       1")
    print(f"colour norm (linear part)  sigma = {colour_norm_singular_values()}  -> 1")
    print(f"resize 1080 -> 600         {r600:.5f}")
    print(f"resize  600 -> 224         {r224:.5f}")
    print(f"ImageNet Normalize (1/std) {inv_std:.5f}")
    print(f"raw -> BRONet input        {r600 * r224:.5f}")
    print(f"raw -> paper-model input   {r600 * r224 * inv_std:.5f}")
    print(f"600 file -> BRONet input   {r224:.5f}")
