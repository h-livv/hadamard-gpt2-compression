"""
GPT-2 plumbing shared by every experiment: reading and writing MLP weights,
the WikiText-2 test perplexity, and the calibration Gram matrices.

None of this is the method being studied (that is in hadamard.py); it is the test bench.
"""

import random

import numpy as np
import torch
from datasets import load_dataset

LAYERS = range(12)               # GPT-2 small has 12 transformer blocks
MATRICES = ["c_fc", "c_proj"]    # the two linear maps inside each block's MLP
STRIDE = 512                     # sliding-window stride of the perplexity guide
SEED = 42                        # paper §5
N_CALIBRATION = 64               # calibration sequences, from the TRAIN split
CALIBRATION_LENGTH = 512         # tokens per calibration sequence
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_weight(model, layer, name):
    """Return one MLP weight matrix of GPT-2 in the usual (out, in) layout.

    THE ONE PLACE (with set_weight) THAT DEALS WITH GPT-2's TRANSPOSED STORAGE.
    GPT-2's MLP layers are `Conv1D` modules, not `nn.Linear`. Conv1D computes
    y = x @ weight + bias with weight stored as (in_features, out_features),
    the transpose of nn.Linear's (out_features, in_features). We transpose
    here so that the rest of the code sees an ordinary linear-layer matrix.

    Args:    model (GPT2LMHeadModel), layer in 0..11, name "c_fc" or "c_proj".
    Returns: X (out, in): (3072, 768) for c_fc, (768, 3072) for c_proj. A copy.
    Source:  transformers/pytorch_utils.py, class Conv1D
             (`self.weight = nn.Parameter(torch.empty(nx, nf))`, nx = inputs,
             nf = outputs; forward is `torch.addmm(self.bias, x, self.weight)`),
             https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py
             transformers/models/gpt2/modeling_gpt2.py, class GPT2MLP (c_fc, c_proj).
    """
    conv1d = getattr(model.transformer.h[layer].mlp, name)
    return conv1d.weight.detach().T.clone()                  # (out, in)


def set_weight(model, layer, name, X):
    """Copy a dense (out, in) matrix into a GPT-2 MLP layer. Inverse of get_weight.

    Args:    X (out, in), same layout as get_weight returns.
    Source:  as get_weight.
    """
    conv1d = getattr(model.transformer.h[layer].mlp, name)
    with torch.no_grad():
        conv1d.weight.copy_(X.T)                             # stored as (in, out)


def load_test_tokens(tokenizer):
    """WikiText-2 test set as one long row of token ids.

    Returns: input_ids (1, seq_len).
    Source:  https://huggingface.co/docs/transformers/perplexity
             (the `load_dataset` / `"\\n\\n".join(test["text"])` block, unchanged).
    """
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    return encodings.input_ids                               # (1, seq_len)


def perplexity(model, input_ids):
    """Perplexity = exp(average negative log-likelihood per token), strided sliding window.

    Each window is up to 1024 tokens and windows start every STRIDE tokens.
    Only the tokens not already scored by the previous window count towards
    the loss (label -100 means "ignore"); the earlier ones are context only.

    Args:    model (GPT2LMHeadModel), input_ids (1, seq_len).
    Returns: float.
    Source:  https://huggingface.co/docs/transformers/perplexity, the loop
             under "Example: Calculating perplexity with GPT-2", stride 512.
    Differs: no tqdm progress bar and no accelerate (a plain device string);
             the guide's example uses gpt2-large, we use gpt2 (small).
    """
    max_length = model.config.n_positions                    # 1024 for GPT-2
    seq_len = input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, STRIDE):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc                     # may differ from STRIDE on the last loop
        window = input_ids[:, begin_loc:end_loc].to(DEVICE)  # (1, <=1024)
        target_ids = window.clone()                          # (1, <=1024)
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            # loss = mean cross-entropy over the labels that are not -100
            neg_log_likelihood = model(window, labels=target_ids).loss

        # The model shifts labels left by one internally, so one fewer token
        # is actually scored than is unmasked.
        num_valid_tokens = (target_ids != -100).sum().item()
        num_loss_tokens = num_valid_tokens - target_ids.size(0)
        nll_sum += neg_log_likelihood.item() * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return float(np.exp(nll_sum / n_tokens))


def load_calibration_tokens(tokenizer):
    """N_CALIBRATION random windows of CALIBRATION_LENGTH tokens from the WikiText-2 TRAIN split.

    The test split is never touched here, so the diagnostics cannot leak
    into the perplexity measurement.

    Returns: calibration_ids (N_CALIBRATION, CALIBRATION_LENGTH) = (64, 512).
    Source:  SVD-LLM, utils/data_utils.py, `get_calib_train_data`
             (https://github.com/AIoT-MLSys-Lab/SVD-LLM): `random.seed(seed)`,
             the "wikitext-2-raw-v1" train split joined with "\\n\\n", and
             `random.randint(0, length - seqlen - 1)` for each window start.
    Differs: SVD-LLM draws a random CHARACTER offset and tokenises from there;
             we tokenise the whole split once and draw a random TOKEN offset,
             which gives windows of exactly 512 tokens without a retry loop.
             Seed 42 (the paper's) and 64 x 512 are this project's settings.
    """
    random.seed(SEED)
    train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    train_ids = tokenizer("\n\n".join(train["text"]), return_tensors="pt").input_ids   # (1, n_train_tokens)
    n_train_tokens = train_ids.size(1)
    windows = []
    for _ in range(N_CALIBRATION):
        i = random.randint(0, n_train_tokens - CALIBRATION_LENGTH - 1)
        windows.append(train_ids[:, i:i + CALIBRATION_LENGTH])  # (1, 512)
    return torch.cat(windows, dim=0)                             # (64, 512)


def add_to_gram(module, args, output):
    """Forward hook: add X^T X of this call's input to `module.G`.

    PyTorch calls a forward hook as hook(module, args, output) after every
    forward pass of `module`; args[0] is the layer's input.

    Args:    module (Conv1D with attribute G (in, in)), args[0] (batch, seq, in).
    Source:  torch.nn.Module.register_forward_hook,
             https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
             SVD-LLM, SVDLLM.py, `profle_svdllm`, inner function `hook`
             (`adds = torch.matmul(inp.transpose(1,2), inp)`, summed into
             `module.raw_scaling_diag_matrix`).
    Differs: we flatten batch and sequence first (same sum) and accumulate in
             float64, because G sums 32,768 float32 outer products.
    """
    X = args[0].detach().double().reshape(-1, args[0].size(-1))  # (batch * seq, in)
    module.G += X.T @ X                                          # (in, in)


def collect_gram_matrices(model, calibration_ids):
    """Gram matrix G = X^T X of the inputs to every MLP matrix, over all calibration tokens.

    X is the (tokens, in) matrix whose rows are the inputs the layer sees.
    Only G is kept, so memory does not grow with the number of tokens.

    Hook points, from GPT2MLP.forward in transformers/models/gpt2/modeling_gpt2.py:
        hidden_states = self.c_fc(hidden_states)     <- input of c_fc: 768 numbers per token
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)   <- input of c_proj: 3072 numbers per token
    A forward hook on the c_fc / c_proj module receives exactly that input.

    Args:    model (GPT2LMHeadModel, original weights), calibration_ids (N, L).
    Returns: dict {(layer, name): G} with G (768, 768) for c_fc, (3072, 3072) for c_proj, float64.
    Source:  as add_to_gram; https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
    """
    handles = []
    for layer in LAYERS:
        for name in MATRICES:
            conv1d = getattr(model.transformer.h[layer].mlp, name)
            conv1d.G = torch.zeros(conv1d.nx, conv1d.nx, dtype=torch.float64, device=DEVICE)   # (in, in)
            handles.append(conv1d.register_forward_hook(add_to_gram))

    with torch.no_grad():
        for sequence in calibration_ids:                         # one (L,) sequence at a time
            model(sequence[None, :].to(DEVICE))

    for handle in handles:
        handle.remove()                                          # the model is back to normal
    grams = {}
    for layer in LAYERS:
        for name in MATRICES:
            conv1d = getattr(model.transformer.h[layer].mlp, name)
            grams[(layer, name)] = conv1d.G                      # (in, in)
            del conv1d.G
    return grams
