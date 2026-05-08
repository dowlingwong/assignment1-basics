# CS336 Assignment 1 Implementation Roadmap

This roadmap summarizes what must be implemented for CS336 Assignment 1 and gives a practical order for completing the project.


## Repository Structure

- `cs336_basics/`: your implementation package.
- `tests/adapters.py`: thin glue between tests and your implementation. Do not put substantive logic here.
- `tests/test_*.py`: autograder-style tests. Do not edit these.
- `cs336_assignment1_basics.pdf`: full handout and written/problem requirements.

## Main Things To Implement

### 1. BPE Tokenizer

Required behavior:

- Byte-level BPE training.
- GPT-2 regex pre-tokenization.
- Special tokens as hard boundaries during training.
- Special tokens preserved as single tokens during encoding.
- Deterministic merge tie-breaking: among equally frequent pairs, choose the lexicographically greatest pair.
- `encode`, `decode`, and `encode_iterable`.

Relevant tests:

- `tests/test_train_bpe.py`
- `tests/test_tokenizer.py`

Likely files:

- `cs336_basics/tokenizer/tokenizer.py`
- `cs336_basics/tokenizer/merge_fn.py`
- `cs336_basics/tokenizer/utils.py`

### 2. Transformer Components

Required modules:

- `Linear`
- `Embedding`
- `RMSNorm`
- `SiLU`
- `SwiGLU`
- RoPE / rotary positional embeddings
- stable softmax
- scaled dot-product attention
- causal multi-head self-attention
- pre-norm Transformer block
- full Transformer language model

Relevant tests:

- `tests/test_model.py`
- `tests/test_nn_utils.py`

Likely files:

- `cs336_basics/modules/linear.py`
- `cs336_basics/modules/embedding.py`
- `cs336_basics/modules/norm.py`
- `cs336_basics/modules/ffn.py`
- `cs336_basics/modules/rope.py`
- `cs336_basics/modules/attention.py`
- `cs336_basics/model.py`

### 3. Training Utilities

Required pieces:

- cross-entropy loss
- random language-model batch sampler
- AdamW optimizer
- cosine learning-rate schedule with warmup
- gradient clipping
- checkpoint save/load

Relevant tests:

- `tests/test_nn_utils.py`
- `tests/test_data.py`
- `tests/test_optimizer.py`
- `tests/test_serialization.py`

Likely files:

- `cs336_basics/utils.py`
- new training/optimizer utility files if you choose to add them

### 4. Training And Generation Scripts

Required capabilities:

- Train a tokenizer on TinyStories.
- Encode datasets to token ID arrays.
- Train a Transformer LM.
- Validate periodically.
- Save and load checkpoints.
- Log train and validation loss over gradient steps and wall-clock time.
- Compute validation loss and perplexity.
- Generate text with temperature and top-p sampling.

Suggested files to add:

- `scripts/train_bpe.py`
- `scripts/encode_dataset.py`
- `scripts/train_lm.py`
- `scripts/generate.py`
- `scripts/evaluate.py`

### 5. Experiments And Writeup

Written/conceptual deliverables:

- Unicode questions.
- BPE tokenizer experiments.
- Tokenizer behavior/compression experiments.
- Transformer resource accounting.
- AdamW resource accounting.

Training deliverables:

- TinyStories baseline training curve.
- Learning-rate sweep.
- Batch-size sweep.
- Generated TinyStories sample.
- RMSNorm removal ablation.
- post-norm vs pre-norm ablation.
- NoPE vs RoPE ablation.
- SwiGLU vs SiLU ablation.
- OpenWebText training curve and generated sample.
- Leaderboard run if required by your course setup.

## Step-By-Step Plan

### Step 0: Fix The Environment

Goal: make the test runner usable.

Checklist:

- Run `uv run pytest`.
- If `uv.lock` fails to parse, regenerate or repair the environment before implementing code.
- Confirm dependencies install successfully.

Expected result:

- Tests run and fail with implementation-related errors, not environment errors.

### Step 1: Wire The Adapters

Goal: make tests call your real code.

Checklist:

- Fill each function in `tests/adapters.py` by constructing/calling your implementation.
- Keep adapters thin.
- Do not implement algorithms inside adapters.

Expected result:

- Tests now fail in your implementation files, which is easier to debug.

### Step 2: Implement Basic NN Modules

Suggested order:

1. `Linear`
2. `Embedding`
3. `RMSNorm`
4. `SiLU`
5. `SwiGLU`

Tests to run:

```sh
uv run pytest -k "test_linear or test_embedding or test_rmsnorm or test_silu or test_swiglu"
```

Debug focus:

- Weight shapes must match the reference state dict.
- Avoid using disallowed `torch.nn`/`torch.nn.functional` helpers.
- RMSNorm should upcast to `float32` for the normalization calculation.

### Step 3: Implement Attention And RoPE

Suggested order:

1. stable softmax
2. scaled dot-product attention
3. RoPE
4. causal multi-head self-attention without RoPE
5. causal multi-head self-attention with RoPE

Tests to run:

```sh
uv run pytest -k "softmax or scaled_dot_product_attention or rope or multihead_self_attention"
```

Debug focus:

- Shape handling for arbitrary batch dimensions.
- Mask convention: `True` means attend, `False` means mask out.
- RoPE must use the head dimension, not the full model dimension.

### Step 4: Implement Transformer Block And LM

Suggested order:

1. pre-norm Transformer block
2. token embedding
3. stack of Transformer blocks
4. final RMSNorm
5. LM head

Tests to run:

```sh
uv run pytest -k "transformer_block or transformer_lm"
```

Debug focus:

- State dict key names and tensor shapes.
- Residual connection order.
- RoPE token positions for truncated inputs.
- Logits shape should be `(batch_size, sequence_length, vocab_size)`.

### Step 5: Implement Training Utilities

Suggested order:

1. cross-entropy
2. batch sampling
3. AdamW
4. cosine LR schedule
5. gradient clipping
6. checkpointing

Tests to run:

```sh
uv run pytest -k "cross_entropy or get_batch or adamw or lr_cosine or gradient_clipping or checkpointing"
```

Debug focus:

- Cross-entropy must be numerically stable.
- Batch labels are inputs shifted by one token.
- AdamW state dict must serialize correctly.
- Checkpointing must restore both model and optimizer state.

### Step 6: Implement BPE Training

Goal: pass the BPE training tests, including speed and special-token behavior.

Tests to run:

```sh
uv run pytest tests/test_train_bpe.py
```

Debug focus:

- Initial vocabulary has 256 byte tokens plus special tokens.
- Special tokens should not contribute to merge statistics.
- Merges do not cross pre-token boundaries.
- Tie-breaking must match the handout exactly.
- The speed test requires an efficient enough implementation.

### Step 7: Implement Tokenizer Encode/Decode

Goal: match GPT-2 tokenizer outputs on the provided fixtures.

Tests to run:

```sh
uv run pytest tests/test_tokenizer.py
```

Debug focus:

- `decode(encode(text)) == text`.
- Special tokens are preserved.
- Overlapping special tokens should prefer longer matches.
- `encode_iterable` should stream rather than load huge files into memory.

### Step 8: Run Full Unit Tests

Goal: establish that implementation-level requirements are passing.

Command:

```sh
uv run pytest
```

Expected result:

- All non-expected-failure tests pass.

### Step 9: Build Dataset Pipeline

Checklist:

- Download TinyStories and OpenWebText.
- Train BPE tokenizer.
- Save vocab and merges.
- Encode train/validation datasets.
- Store token IDs in an efficient format such as `.npy` or `np.memmap`.

Suggested commands/scripts:

- `scripts/train_bpe.py`
- `scripts/encode_dataset.py`

### Step 10: Build Training Loop

Checklist:

- Parse model/training hyperparameters.
- Load memmapped train/validation token arrays.
- Sample batches.
- Forward pass.
- Cross-entropy loss.
- Backward pass.
- Gradient clipping.
- AdamW step.
- LR schedule update.
- Periodic validation.
- Periodic checkpointing.
- Log metrics with step and wall-clock time.

Suggested script:

- `scripts/train_lm.py`

### Step 11: Sanity Check Training

Before long runs:

- Overfit a single minibatch.
- Confirm loss can go near zero.
- Inspect tensor shapes through one full forward pass.
- Monitor weight, activation, and gradient norms.
- Run a tiny training job on CPU/MPS/GPU.

### Step 12: TinyStories Baseline

Checklist:

- Train baseline Transformer LM.
- Track train and validation loss.
- Tune learning rate.
- Tune batch size.
- Save best checkpoint.
- Generate at least 256 tokens.
- Report validation loss/perplexity.

### Step 13: Required Ablations

Run and log:

- remove RMSNorm
- post-norm instead of pre-norm
- NoPE instead of RoPE
- SiLU FFN instead of SwiGLU FFN

For each:

- Keep compute comparable.
- Record learning curve.
- Compare against baseline.
- Write a short interpretation.

### Step 14: OpenWebText Experiment

Checklist:

- Train same architecture/iteration budget on OpenWebText.
- Retune learning rate if necessary.
- Log train/validation loss.
- Generate text.
- Explain why OpenWebText quality differs from TinyStories.

### Step 15: Leaderboard Run

Checklist:

- Stay within the runtime/data constraints.
- Record final validation loss.
- Produce wall-clock-time learning curve.
- Describe modifications.
- Submit to the leaderboard repo if required.

## Suggested Completion Order

1. Environment works.
2. Adapters are wired.
3. Basic NN module tests pass.
4. Attention/RoPE tests pass.
5. Transformer block/LM tests pass.
6. Training utility tests pass.
7. BPE training tests pass.
8. Tokenizer tests pass.
9. Full test suite passes.
10. TinyStories small sanity training succeeds.
11. TinyStories baseline experiments complete.
12. Ablations complete.
13. OpenWebText experiment complete.
14. Leaderboard/writeup complete.

## Useful Test Commands

```sh
uv run pytest -k test_linear
uv run pytest -k test_embedding
uv run pytest -k test_rmsnorm
uv run pytest -k test_swiglu
uv run pytest -k test_rope
uv run pytest -k test_scaled_dot_product_attention
uv run pytest -k test_multihead_self_attention
uv run pytest -k test_transformer_block
uv run pytest -k test_transformer_lm
uv run pytest tests/test_nn_utils.py
uv run pytest tests/test_optimizer.py
uv run pytest tests/test_data.py
uv run pytest tests/test_serialization.py
uv run pytest tests/test_train_bpe.py
uv run pytest tests/test_tokenizer.py
uv run pytest
```

## Debugging Rules Of Thumb

- Pass one small test group at a time.
- Keep tensor shape comments in your scratch notes.
- Compare against the handout equations before changing code.
- Use targeted tests instead of the full suite while iterating.
- For model bugs, first check weight orientation and state dict key mapping.
- For tokenizer bugs, first check pre-token boundaries, special-token handling, and merge tie-breaking.
- For training bugs, first overfit one batch before running real experiments.
