import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch

    # A) random vector inputs

    random_inputs = torch.randn(16, 128, 64)  # (batch, seq_len, dim)

    # B) Structured signal (sine wave etc)
    structured_inputs = (
        torch.sin(torch.linspace(0, 8 * 3.14, 128))
        .unsqueeze(0)
        .unsqueeze(2)
        .repeat(16, 1, 64)
    )

    # c) impulse inputs

    return


@app.cell
def _():
    ## Ordered attention operators to implement (by hand)

    ### 1. **Vanilla scaled dot-product attention**

    # > Baseline. Everything else is a deviation from this.

    # Implement:
    # [
    # \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
    # ]

    # Focus on:

    # * why scaling by √d matters
    # * what softmax is actually doing to gradients
    # * attention entropy over training

    # ---

    ### 2. **Dot-product attention *without* softmax**

    # [
    # (QK^\top)V ;; \text{with normalization of your choice}
    # ]

    # Variants:

    # * divide by sequence length
    # * row-wise ℓ2 normalization
    # * no normalization at all

    # Learning payoff:

    # * isolates *softmax’s role*
    # * shows why attention explodes or collapses

    ### 3. **Cosine-similarity attention**

    # [
    # \text{softmax}\left(\frac{QK^\top}{|Q||K|}\right)V
    # ]

    # Focus on:

    # * scale invariance
    # * effect on training stability
    # * comparison to dot-product scaling

    ### 4. **Linear / kernelized attention**

    # Replace softmax with feature maps:
    # [
    # \phi(Q)(\phi(K)^\top V)
    # ]

    # Start simple:

    # * φ(x) = ReLU(x)
    # * φ(x) = exp(x) (clipped)

    # Learning payoff:

    # * separates *mixing* from *normalization*
    # * reveals why softmax is expensive but expressive

    ### 5. **Local (sliding-window) attention**

    # Mask attention to a fixed radius.

    # Questions to answer:

    # * what breaks first as window shrinks?
    # * does depth compensate for locality?

    # This teaches **inductive bias**.

    ### 6. **Attention with no learned projections**

    # * Q = K = V = input (or fixed linear map)

    # This is brutal but illuminating:

    # * shows how much attention relies on learned geometry
    # * exposes attention as a *routing mechanism*, not magic

    # ---

    return


if __name__ == "__main__":
    app.run()
