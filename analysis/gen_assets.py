"""
Generate analysis assets from a trained ONE-LAYER shakespeare checkpoint.

Produces:
  - analysis/assets/one_layer_eigen.png   (OV-circuit eigenvalues, per head)
  - analysis/assets/positional_attention.png  (preferred relative positions, per head)

These are the two assets that only require a one-layer model. The induction-head
and two-layer-eigenvalue assets need a trained two-layer model.
"""

import os

import matplotlib
matplotlib.use('Agg')  # headless: save PNGs without a display
import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis.utils import (
    get_weights_for_head,
    get_embedding_weights,
    get_ov_eigenvalues,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..'))
ASSETS = os.path.join(HERE, 'assets')

# our small shakespeare one-layer config
CKPT = os.path.join(REPO_ROOT, 'out', 'shakespeare', 'latest_model_2000.pt')
N_HEADS = 4
D_MODEL = 128


def positional_matrix(head_weights):
    """ Same math as utils.positional_attention_for_head, but returns the heatmap. """
    p_e = head_weights['p_e']
    qk = head_weights['w_q'].T @ head_weights['w_k']
    res = p_e @ qk @ p_e.T
    mask = np.triu(np.ones_like(res), k=1)
    res = res * (1 - mask)
    res = torch.softmax(torch.from_numpy(res), dim=0).numpy()
    return res


def main():
    os.makedirs(ASSETS, exist_ok=True)
    weights = torch.load(CKPT, map_location='cpu')

    # extract per-head weights
    head_weights = [
        get_weights_for_head(weights=weights, layer=0, head=h,
                             n_heads=N_HEADS, d_model=D_MODEL, apply_layernorm=False)
        for h in range(N_HEADS)
    ]
    embedding_weights = get_embedding_weights(weights=weights, d_model=D_MODEL,
                                              norm_emb=True, final_layernorm=True)

    # ---- asset 1: OV eigenvalues (polar scatter per head) ----
    n_rows = N_HEADS // 2
    fig, ax = plt.subplots(2, n_rows, subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    for h in range(N_HEADS):
        eigen = get_ov_eigenvalues(wh=head_weights[h], we=embedding_weights)
        z = eigen.real + 1j * eigen.imag
        axis = ax[h // n_rows, h % n_rows]
        axis.scatter(np.angle(z), np.log(np.abs(z)), s=6)
        axis.set_title(f'head {h}')
        axis.set_xticks([])
    fig.suptitle('One-layer OV-circuit eigenvalues')
    fig.tight_layout()
    out1 = os.path.join(ASSETS, 'one_layer_eigen.png')
    fig.savefig(out1, dpi=120)
    plt.close(fig)
    print('wrote', out1)

    # ---- asset 2: positional attention heatmaps (per head) ----
    fig, ax = plt.subplots(2, n_rows, figsize=(8, 8))
    for h in range(N_HEADS):
        res = positional_matrix(head_weights[h])
        axis = ax[h // n_rows, h % n_rows]
        im = axis.imshow(res, aspect='auto')
        axis.set_title(f'head {h}')
    fig.suptitle('Preferred relative positions (QK circuit)')
    fig.tight_layout()
    out2 = os.path.join(ASSETS, 'positional_attention.png')
    fig.savefig(out2, dpi=120)
    plt.close(fig)
    print('wrote', out2)


if __name__ == '__main__':
    main()
