# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Small Language Model (SLM) implementation based on the GPT architecture, designed to be trained on the TinyStories dataset. The model is intentionally lightweight and educational, using ~6 transformer layers with 384 embedding dimensions.

## Architecture

### Model Components (in `Small_Language_Model_Building (1).py`)

**Core Architecture Classes:**

1. **`GPTConfig`** (dataclass): Configuration for model hyperparameters
   - `block_size`: Context window (default: 128)
   - `vocab_size`: 50257 (GPT-2 tokenizer vocabulary)
   - `n_layer`: Number of transformer blocks (default: 6)
   - `n_head`: Attention heads (default: 6)
   - `n_embd`: Embedding dimension (default: 384)
   - `dropout`: Regularization (default: 0.1)
   - `bias`: Whether to use bias in linear layers

2. **`GPT`** (nn.Module): Main model class
   - Token embeddings (`wte`) and positional embeddings (`wpe`)
   - Stack of `Block` layers
   - Final layer norm and language modeling head
   - **Weight tying**: `wte` weights are shared with `lm_head`
   - Supports Flash Attention when available (PyTorch 2.0+)

3. **`Block`** (nn.Module): Transformer block with:
   - Pre-normalization LayerNorm
   - `CausalSelfAttention` with residual connection
   - `MLP` (feedforward) with residual connection

4. **`CausalSelfAttention`**: Multi-head self-attention
   - Splits into Q, K, V via single linear projection (`c_attn`)
   - Uses Flash Attention (`F.scaled_dot_product_attention`) when available
   - Falls back to manual attention computation with causal masking

5. **`MLP`**: Feedforward network with GELU activation
   - Expands to 4× embedding dimension (`c_fc`) then projects back (`c_proj`)

6. **`LayerNorm`**: Custom layer normalization implementation

### Data Pipeline

**Tokenization:**
- Uses `tiktoken` with GPT-2 encoding ("gpt2")
- Processes TinyStories dataset via `datasets` library
- Stores tokenized data as memory-mapped binary files (`train.bin`, `validation.bin`)
- Uses `np.uint16` dtype (2 bytes per token)

**Data Loading:**
- `get_batch(split)` function loads random subsequences
- Creates input/target pairs with offset-by-1 (next token prediction)
- Supports memory pinning for GPU async transfer
- Uses `np.memmap` for disk-based data access

### Training Configuration

**Key Hyperparameters:**
```python
learning_rate = 1e-4
max_iters = 10000
warmup_steps = 1000
min_lr = 5e-4
batch_size = 32
block_size = 128
gradient_accumulation_steps = 32
```

**Training Features:**
- **Mixed Precision**: Uses `bfloat16` when available, else `float16`
- **Gradient Scaling**: `torch.cuda.amp.GradScaler` for FP16 stability
- **Learning Rate Scheduling**: Linear warmup → Cosine annealing
- **Optimizer**: AdamW with weight decay (0.1), betas=(0.9, 0.95)
- **Gradient Clipping**: Max norm 0.5
- **Checkpointing**: Saves best model based on validation loss

## Common Development Commands

**Install Dependencies:**
```bash
pip install torch datasets tiktoken tqdm matplotlib
```

**Run Training:**
```bash
python "Small_Language_Model_Building (1).py"
```

**Run Jupyter Notebook:**
```bash
jupyter notebook "Small_Language_Model_Building (1).ipynb"
```

**Generate Text (after training):**
```python
# Load saved model
model = GPT(config)
model.load_state_dict(torch.load("best_model_params.pt"))

# Generate
sentence = "Once upon a time"
context = torch.tensor(encoding.encode_ordinary(sentence)).unsqueeze(0)
output = model.generate(context, max_new_tokens=200)
print(encoding.decode(output.squeeze().tolist()))
```

## File Structure

- `Small_Language_Model_Building (1).py` - Main training script with full model implementation
- `Small_Language_Model_Building (1).ipynb` - Jupyter notebook version of the training script
- `SLM_Instruction-tuned_model.ipynb` - Instruction tuning variant (appears to be work-in-progress)
- `train.bin` / `validation.bin` - Memory-mapped tokenized datasets (generated during first run)
- `best_model_params.pt` - Saved model checkpoint with lowest validation loss

## Important Implementation Details

**Pre-tokenization Strategy:**
The dataset is pre-tokenized and stored as `.bin` files to avoid repeated tokenization and reduce RAM usage during training. Check for file existence before re-tokenizing:
```python
if not os.path.exists("train.bin"):
    # Run tokenization pipeline
```

**Context Size in Training:**
The model is trained with `block_size=128`, meaning it sees 128 tokens and predicts the 129th. The sliding window approach creates multiple training examples from each sequence.

**Device Handling:**
The code automatically detects CUDA and uses:
- `torch.amp.autocast` for mixed precision on GPU
- `pin_memory()` and `non_blocking=True` for async data transfer
- Falls back to CPU if CUDA unavailable

**Loss Calculation:**
Uses cross-entropy loss over the entire vocabulary. For a batch, the loss is the average over all token positions in the batch.

**Generation Parameters:**
- `temperature`: Controls randomness (1.0 = default)
- `top_k`: Limits sampling to top-k tokens (None = use full vocabulary)

## Dependencies

Core dependencies (no formal requirements.txt exists):
- `torch` (2.0+ recommended for Flash Attention)
- `datasets` (HuggingFace)
- `tiktoken` (OpenAI's BPE tokenizer)
- `tqdm` (progress bars)
- `numpy`
- `matplotlib` (for loss plotting)
