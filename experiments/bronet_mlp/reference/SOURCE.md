# Reference scripts

Standalone scripts, kept as they are; they are **not** imported by the
package. They expect to sit in one folder `pretrained/` together with the BRONet
clone (`pretrained/BRONet/`) and the checkpoint (`pretrained/bronet_imagenet/`). `bronet.py` reimplements the functions it needs
(`sigma_range`, `conv_fft_bound`, the checkpointed `features()` as
`phi_with_grad`).

| file | location it expects | what it is |
|---|---|---|
| `bronet_lipschitz.py` | `pretrained/bronet_lipschitz.py` | BRONet loading, per-layer checkpointed forward, rigorous Lipschitz audit, empirical lower bound |
| `lipschitz.json` | `pretrained/bronet_imagenet/` | output of the audit above on `imagenet-14x588_399.pth` (official 8.1305, estimate 8.1046, rigorous 8.4031, empirical lower 6.53); reference values for the tests |
| `preprocessing_lipschitz.py` | `pretrained/preprocessing_lipschitz.py` | operator norms of the preprocessing resizes (600 -> 224 = 0.37458) |

BRONet itself: https://github.com/ntuaislab/BRONet, commit 9d59c9b, MIT licence.
Checkpoint: https://huggingface.co/pinhank121/BRONet_ImageNet (`imagenet-14x588_399.pth`,
sha256 `e8521d477bdeb80b0a623b09a7ce3025b874f21984e50686e869ab61694ca95f`).
