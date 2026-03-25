import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from dataclasses import dataclass
import tiktoken

# Set page config
st.set_page_config(
    page_title="TinyStories Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    .chat-message {
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        max-width: 80%;
        word-wrap: break-word;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin-left: auto;
        margin-right: 0;
        text-align: right;
    }
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        margin-right: auto;
        margin-left: 0;
    }
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 25px;
        padding: 15px 25px;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 15px 30px;
        border: none;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
    }
    .sidebar-content {
        background-color: rgba(0, 0, 0, 0.2);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .model-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .title-container {
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }
    .chat-container {
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 20px;
        padding: 20px;
        margin: 20px 0;
        min-height: 400px;
        max-height: 500px;
        overflow-y: auto;
    }
    .typing-indicator {
        display: flex;
        align-items: center;
        margin: 10px 0;
    }
    .typing-indicator span {
        width: 8px;
        height: 8px;
        background-color: #fff;
        border-radius: 50%;
        display: inline-block;
        margin: 0 2px;
        animation: bounce 1.4s infinite ease-in-out both;
    }
    .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
    .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# Model Architecture Classes
@dataclass
class GPTConfig:
    block_size: int = 128
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                       .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                    dropout_p=self.dropout if self.training else 0,
                                                                    is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Load model function with caching
@st.cache_resource
def load_model(model_path):
    config = GPTConfig()
    model = GPT(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # Handle different checkpoint formats
        state_dict = None
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        elif isinstance(checkpoint, GPT):
            state_dict = checkpoint.state_dict()

        if state_dict is not None:
            # Fix key name mismatches (ln_1 -> ln1, ln_2 -> ln2)
            fixed_state_dict = {}
            for key, value in state_dict.items():
                new_key = key
                if 'ln_1' in key:
                    new_key = key.replace('ln_1', 'ln1')
                elif 'ln_2' in key:
                    new_key = key.replace('ln_2', 'ln2')
                fixed_state_dict[new_key] = value

            # Load with strict=False to ignore mismatched keys
            model.load_state_dict(fixed_state_dict, strict=False)

        model = model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, device

# Sidebar for model selection and settings
with st.sidebar:
    st.title("⚙️ Settings")

    # Model Selection
    st.subheader("Model Selection")
    model_files = {
        "Model Weights": "model_weights.pt",
        "Full Model": "full_model.pt",
        "Checkpoint": "checkpoint.pt"
    }

    model_choice = st.radio("Select model file:", list(model_files.keys()))
    selected_model_path = model_files[model_choice]

    if not os.path.exists(selected_model_path):
        st.warning(f"⚠️ {selected_model_path} not found!")
    else:
        st.success(f"✅ {selected_model_path} found")

    st.markdown("---")

    # Generation Parameters
    st.subheader("Generation Parameters")
    temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1,
                          help="Higher = more creative, Lower = more focused")
    top_k = st.slider("Top-K", 1, 100, 40,
                      help="Limits vocabulary for sampling")
    max_tokens = st.slider("Max Response Length", 50, 500, 150, 10,
                          help="Maximum tokens to generate")

    st.markdown("---")

    # Chat Settings
    st.subheader("Chat Settings")
    clear_chat = st.button("🗑️ Clear Chat History", use_container_width=True)

    st.markdown("---")
    st.markdown("### About")
    st.info("This chatbot is powered by a custom Small Language Model trained on TinyStories dataset.")

# Initialize tokenizer
@st.cache_resource
def get_tokenizer():
    return tiktoken.get_encoding("gpt2")

encoding = get_tokenizer()

# Main app layout
st.markdown("""
<div class="title-container">
    <h1>🤖 TinyStories Chatbot</h1>
    <p>Your AI storytelling companion powered by a custom Small Language Model</p>
</div>
""", unsafe_allow_html=True)

# Load model with session state
if "model" not in st.session_state or st.session_state.get('loaded_model_path') != selected_model_path:
    if os.path.exists(selected_model_path):
        with st.spinner(f"Loading {model_choice}..."):
            model, device = load_model(selected_model_path)
        if model:
            st.session_state.model = model
            st.session_state.device = device
            st.session_state.loaded_model_path = selected_model_path
        else:
            st.error("Failed to load model. Please check the model file.")
            st.stop()
    else:
        st.error(f"❌ Model file not found: {selected_model_path}")
        st.stop()

model = st.session_state.model
device = st.session_state.device

# Initialize chat history
if clear_chat or "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your TinyStories AI companion. I love creating stories and having conversations. What would you like to talk about?"}
    ]

# Display chat messages with Streamlit Chat elements
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input using modern chat input
prompt = st.chat_input("Type your message...")

if prompt:
    # Display user input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate model response
    with st.chat_message("assistant"):
        with st.spinner("AI is thinking..."):
            # Prepare context
            context = ""
            for msg in st.session_state.messages[-3:]:  
                if msg["role"] == "user":
                    context += f"User: {msg['content']}\n"
                else:
                    context += f"AI: {msg['content']}\n"
            context += "AI:"

            # Encode input
            input_ids = torch.tensor(encoding.encode_ordinary(context), dtype=torch.long).unsqueeze(0).to(device)

            # Generate
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k
                )

            # Decode full sequence
            full_output = encoding.decode(output[0].tolist())
            prompt_text = encoding.decode(input_ids[0].tolist())
            
            # Extract new string
            response_text = full_output[len(prompt_text):].strip()

            # Clean up response stopping exactly on delimiters
            stop_tokens = ["User:", "Human:", "AI:", "\n\n"]
            for token in stop_tokens:
                if token in response_text:
                    response_text = response_text[:response_text.index(token)].strip()

            st.write(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.markdown("📊 Model: SLM (6 layers, 384 dim)")
with footer_col2:
    st.markdown(f"🔤 Vocab Size: {GPTConfig().vocab_size:,}")
with footer_col3:
    st.markdown(f"📏 Context: {GPTConfig().block_size} tokens")
