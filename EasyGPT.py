import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_heads, block_size, dropout):
        super().__init__()
        self.n_embd = n_embd 
        self.n_heads = n_heads
        self.head_size = n_embd // n_heads
        self.block_size = block_size
        self.attn = nn.Linear(n_embd, 3*n_embd, bias = False)
        self.proj = nn.Linear(n_embd, n_embd, bias = False)
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.attn_dropout = nn.Dropout(dropout)
        self.resi_dropout = nn.Dropout(dropout)
        self.dropout = dropout 
    def forward(self, x):
        B,T,C = x.shape 
        q,k,v = self.attn(x).split(self.n_embd, dim = -1)
        q = q.view(B,T, self.n_heads, self.head_size).transpose(1,2)
        k = k.view(B,T, self.n_heads, self.head_size).transpose(1,2)
        v = v.view(B,T, self.n_heads, self.head_size).transpose(1,2)
        if self.flash:
            out = F.scaled_dot_product_attention(q,k,v, is_causal = True, dropout_p = self.dropout if self.training else 0.0)
        else:
            wei = q @ k.transpose(-2,-1) / math.sqrt(k.size(-1))
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim = -1)
            wei = self.attn_dropout(wei)
            out = wei @ v
        out = out.transpose(1,2).contiguous().view(B,T,C)
        out = self.resi_dropout(self.proj(out)) 
        return out 

class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd, bias = False),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd, bias = False),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = self.model(x)
        return x 
class Block(nn.Module):
    def __init__(self, n_embd, n_heads, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd) 
        self.mlp = MLP(n_embd, dropout)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x)) 
        return x 
class MiniGPT(nn.Module):
    def __init__(self, n_embd, n_heads, block_size, dropout, vocab_size, n_layers):
        super().__init__()
        self.n_embd = n_embd 
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.tok_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Parameter(torch.randn(1, block_size, n_embd))
        nn.init.normal_(self.pos_embedding, mean = 0.0, std = 0.02)
        self.blocks = nn.ModuleList([Block(n_embd, n_heads, block_size, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias = False)
    def forward(self, idx):
        B,T = idx.shape 
        tok_embd = self.tok_embedding(idx)
        pos_embd = self.pos_embedding[:, :T, :] 
        x = tok_embd + pos_embd 
        for b in self.blocks:
            x = b(x) 
        x = self.ln_f(x)
        logits = self.lm_head(x) 
        loss = None 
        logits_s = logits[:,:-1, :].reshape(-1, self.vocab_size)
        targets = idx[:, 1:].reshape(-1) 
        loss = F.cross_entropy(logits_s, targets) 
        print(loss.item())
        return logits, loss 
        
def generate(idx, model, max_new_tokens):
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_in = idx[:, -model.block_size:] 
            logits,_ = model(idx_in) # b,t,c
            logits = logits[:,-1,:] # b,c
            new_token = F.softmax(logits, dim = -1) # b,c
            new_token = torch.multinomial(new_token, num_samples = 1) # b,1
            idx = torch.cat([idx, new_token], dim = 1)
    return idx 


# ── helpers ────────────────────────────────────────────────────────────────────

def make_model(vocab_size=50, block_size=16, n_embd=32, n_heads=4,
               n_layers=2, dropout=0.0):
    return MiniGPT(n_embd=n_embd, n_heads=n_heads, block_size=block_size,
                   dropout=dropout, vocab_size=vocab_size, n_layers=n_layers)

def rand_idx(B=2, T=8, vocab_size=50):
    return torch.randint(0, vocab_size, (B, T))


# ── tests ──────────────────────────────────────────────────────────────────────

def test_forward_shapes():
    """logits shape and loss is a scalar."""
    print("test_forward_shapes ...", end=" ")
    model = make_model()
    idx = rand_idx()
    logits, loss = model(idx)

    B, T = idx.shape
    assert logits.shape == (B, T, 50), f"logits shape: {logits.shape}"
    assert loss.ndim == 0, "loss should be a scalar"
    assert loss.item() > 0
    print("OK")


def test_loss_decreases():
    """Loss should decrease over a few gradient steps."""
    print("test_loss_decreases ...", end=" ")
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    idx = rand_idx(B=4, T=12)

    losses = []
    for _ in range(20):
        model.train()
        optimizer.zero_grad()
        _, loss = model(idx)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print("OK")


def test_causal_mask():
    """Changing a future token must not affect earlier positions."""
    print("test_causal_mask ...", end=" ")
    model = make_model()
    model.eval()
    idx = rand_idx(B=1, T=8)

    with torch.no_grad():
        logits_orig, _ = model(idx)

        idx_mod = idx.clone()
        idx_mod[0, -1] = (idx[0, -1] + 1) % 50   # change last token
        logits_mod, _ = model(idx_mod)

    # positions 0..T-2 should be identical (causal)
    assert torch.allclose(logits_orig[0, :-1], logits_mod[0, :-1], atol=1e-5), \
        "causal mask broken: future token affected earlier positions"
    print("OK")


def test_generate_shape():
    """generate() should extend sequence by exactly max_new_tokens."""
    print("test_generate_shape ...", end=" ")
    model = make_model()
    idx = rand_idx(B=1, T=4)
    max_new = 6

    out = generate(idx, model, max_new_tokens=max_new)
    assert out.shape == (1, 4 + max_new), f"generate shape: {out.shape}"
    # first 4 tokens must be unchanged
    assert torch.equal(out[:, :4], idx), "generate modified the prompt tokens"
    print("OK")


def test_generate_tokens_in_vocab():
    """All generated tokens must be valid vocab indices."""
    print("test_generate_tokens_in_vocab ...", end=" ")
    vocab_size = 50
    model = make_model(vocab_size=vocab_size)
    idx = rand_idx(B=1, T=4, vocab_size=vocab_size)
    out = generate(idx, model, max_new_tokens=10)

    new_tokens = out[0, 4:]
    assert new_tokens.min() >= 0 and new_tokens.max() < vocab_size, \
        f"out-of-vocab token: min={new_tokens.min()}, max={new_tokens.max()}"
    print("OK")


def test_generate_long_context():
    """generate() should handle prompts longer than block_size via cropping."""
    print("test_generate_long_context ...", end=" ")
    block_size = 16
    model = make_model(block_size=block_size)
    idx = rand_idx(B=1, T=block_size + 4)   # longer than block_size
    out = generate(idx, model, max_new_tokens=3)
    assert out.shape[1] == block_size + 4 + 3
    print("OK")


def test_block_residual():
    """Block output should differ from input (residual connection is active)."""
    print("test_block_residual ...", end=" ")
    block = Block(n_embd=32, n_heads=4, block_size=16, dropout=0.0)
    block.eval()
    x = torch.randn(2, 8, 32)
    with torch.no_grad():
        out = block(x)
    assert not torch.allclose(out, x), "Block output identical to input — residual may be broken"
    print("OK")


def test_dropout_train_vs_eval():
    """Model output should be deterministic in eval mode."""
    print("test_dropout_train_vs_eval ...", end=" ")
    model = make_model(dropout=0.5)
    idx = rand_idx(B=1, T=8)

    model.eval()
    with torch.no_grad():
        logits1, _ = model(idx)
        logits2, _ = model(idx)
    assert torch.allclose(logits1, logits2), "eval mode not deterministic"
    print("OK")


# ── run all ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_forward_shapes()
    test_loss_decreases()
    test_causal_mask()
    test_generate_shape()
    test_generate_tokens_in_vocab()
    test_generate_long_context()
    test_block_residual()
    test_dropout_train_vs_eval()
    print("All tests passed.")

        
        