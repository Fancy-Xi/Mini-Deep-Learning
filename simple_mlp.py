import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ─────────────────────────────────────────
# Approach 1: Manual (raw tensors)
# ─────────────────────────────────────────

def mlp(x, y, n_iter, n_hidden, n_class, lr):
    x = torch.as_tensor(x, dtype = torch.float32) # n,x
    y = torch.as_tensor(y, dtype = torch.long)    # n 
    
    N,d = x.shape
    g = torch.Generator().manual_seed(28)
    w1 = torch.randn(d, n_hidden, generator = g) / math.sqrt(d)
    w2 = torch.randn(n_hidden, n_class, generator = g) / math.sqrt(n_hidden)
    w1.requires_grad = True
    w2.requires_grad = True 
    b1 = torch.zeros(n_hidden, requires_grad = True)
    b2 = torch.zeros(n_class, requires_grad = True)

    parameters = [w1, w2, b1, b2]

    for _ in range(n_iter):
        for p in parameters:
            p.grad = None 

        logits = x @ w1 + b1
        logits = F.gelu(logits)
        logits = logits @ w2 + b2 
        loss = F.cross_entropy(logits, y)

        loss.backward()
        with torch.no_grad():
            for p in parameters:
                p.data -= lr * p.grad 
    return parameters 

def predict(x, parameters):
    x = torch.as_tensor(x, dtype = torch.float32)
    [w1, w2, b1, b2] = parameters
    with torch.no_grad():
        logits = x @ w1 + b1
        logits = F.gelu(logits)
        logits = logits @ w2 + b2
        prob = F.softmax(logits, dim = -1) # n,c
        pred = torch.argmax(prob, dim = -1) # n
        rows = torch.arange(len(x))
        prob = prob[rows, pred] 
    return pred, prob


# ─────────────────────────────────────────
# Approach 2: nn.Module + AdamW
# ─────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_class, dropout):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, n_hidden, bias = False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_class, bias = False)
        )
    def forward(self, x):
        out = self.model(x)
        return out 
        
def mlp_train(x, y, n_hidden, lr, n_iter, n_class, dropout):
    x = torch.as_tensor(x, dtype = torch.float32)
    y = torch.as_tensor(y, dtype = torch.long)
    N,d = x.shape
    model = MLP(d, n_hidden, n_class, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = 0.01)
    for _ in range(n_iter):
        model.train()
        optimizer.zero_grad(set_to_none = True)
        logits = model(x)
        loss = F.cross_entropy(logits, y) 
        loss.backward()
        optimizer.step()
    return model 
def predict_auto(x, model):
    x = torch.as_tensor(x, dtype = torch.float32)
    with torch.no_grad():
        model.eval() 
        logits = model(x)
        prob = F.softmax(logits, dim = -1)
        pred = torch.argmax(prob, dim = -1)
        rows = torch.arange(len(x))
        prob = prob[rows, pred]
    return pred, prob
        
# ─────────────────────────────────────────
# data
# ─────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def make_data(n_samples=500, n_features=10, n_classes=3, random_state=42):
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_classes=n_classes,
        n_informative=6, random_state=random_state
    )
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=random_state)


def accuracy(pred_tensor, y_np):
    return (pred_tensor.numpy() == y_np).mean()


# ─────────────────────────────────────────
# Tests
# ─────────────────────────────────────────

def test_manual(X_train, X_test, y_train, y_test):
    print("=" * 40)
    print("Approach 1: Manual MLP")
    print("=" * 40)
    params = mlp(X_train, y_train, n_iter=100, n_hidden=32, n_class=3, lr=0.05)
    pred, prob = predict(X_test, params)

    acc = accuracy(pred, y_test)
    print(f"Test accuracy:       {acc:.4f}")
    print(f"Predictions (first 10): {pred[:10].numpy()}")
    print(f"Confidences (first 10): {prob[:10].numpy().round(3)}")

    # Shape checks
    assert pred.shape == (len(y_test),), f"pred shape mismatch: {pred.shape}"
    assert prob.shape == (len(y_test),), f"prob shape mismatch: {prob.shape}"
    assert prob.min() >= 0 and prob.max() <= 1, "prob out of [0,1]"
    assert acc > 0.5, f"accuracy too low: {acc:.4f}"
    print("All assertions passed.")


def test_auto(X_train, X_test, y_train, y_test):
    print("=" * 40)
    print("Approach 2: nn.Module MLP (AdamW)")
    print("=" * 40)
    model = mlp_train(X_train, y_train, n_hidden=100, lr=0.05, n_iter=100, n_class=3, dropout=0.1)
    pred, prob = predict_auto(X_test, model)

    acc = accuracy(pred, y_test)
    print(f"Test accuracy:       {acc:.4f}")
    print(f"Predictions (first 10): {pred[:10].numpy()}")
    print(f"Confidences (first 10): {prob[:10].numpy().round(3)}")

    # Shape checks
    assert pred.shape == (len(y_test),), f"pred shape mismatch: {pred.shape}"
    assert prob.shape == (len(y_test),), f"prob shape mismatch: {prob.shape}"
    assert prob.min() >= 0 and prob.max() <= 1, "prob out of [0,1]"
    assert acc > 0.5, f"accuracy too low: {acc:.4f}"

    # Confirm model is in eval mode after predict_auto
    assert not model.training, "model should be in eval mode after predict_auto"
    print("All assertions passed.")


def test_output_shapes_only():
    """Minimal smoke test — no training, just checks forward pass shapes."""
    print("=" * 40)
    print("Smoke test: output shapes")
    print("=" * 40)
    X = torch.randn(8, 10)
    y = torch.randint(0, 3, (8,))

    # Manual
    params = mlp(X.numpy(), y.numpy(), n_iter=1, n_hidden=16, n_class=3, lr=0.01)
    pred, prob = predict(X.numpy(), params)
    assert pred.shape == (8,)
    assert prob.shape == (8,)

    # Auto
    model = mlp_train(X.numpy(), y.numpy(), n_hidden=16, lr=1e-3, n_iter=1, n_class=3, dropout=0.0)
    pred2, prob2 = predict_auto(X.numpy(), model)
    assert pred2.shape == (8,)
    assert prob2.shape == (8,)

    print("Smoke test passed.")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = make_data()
    test_output_shapes_only()
    test_manual(X_train, X_test, y_train, y_test)
    test_auto(X_train, X_test, y_train, y_test)
        
    