"""
super_qai_qml_cybersec.py

Hybrid classical-quantum pipeline for cybersecurity anomaly detection.
Combines a classical deep model (PyTorch) with a small variational quantum circuit
(using PennyLane) as a quantum layer (QAI-QML hybrid).

This is a proof-of-concept educational example — not production-grade.

Requirements:
  - Python 3.8+
  - torch
  - pennylane
  - pennylane-lightning (optional, for faster simulation)
  - scikit-learn
  - pandas, numpy

Install example:
  pip install torch pennylane pennylane-lightning scikit-learn pandas numpy

Usage:
  - If you have a labeled cybersecurity dataset (e.g., NSL-KDD features), point
    the loader to it. Otherwise synthetic data will be generated.
  - Run: python super_qai_qml_cybersec.py

Notes on architecture:
  - Feature preprocessing (PCA / Scaling)
  - Classical encoder (MLP) -> latent vector
  - Quantum variational layer (PennyLane) consumes small latent vector
  - Outputs from classical and quantum parts are concatenated and fed to final MLP

This file keeps things parameterizable so you can swap dataset/models or
experiment with different quantum circuit ansatzes.

"""

import os
import math
import random
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

import pennylane as qml
from pennylane import numpy as qnp

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_FEATURES = 20  # after feature selection/encoding (toy default)
CLASSICAL_LATENT = 8
QUANTUM_QUBITS = 4  # keep small for simulators
QUANTUM_INPUT_DIM = QUANTUM_QUBITS  # how many features the quantum circuit will take
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3

def load_or_generate_data(path: str = None, n_features: int = N_FEATURES, n_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """
    If path points to a CSV with features+label column named 'label', load it.
    Otherwise generate synthetic data: two classes (normal/anomaly) with
    different Gaussian distributions. This is only for demonstration.
    """
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        if 'label' not in df.columns:
            raise ValueError("CSV must have a 'label' column for supervised training")
        X = df.drop(columns=['label']).values
        y = df['label'].values
        return X, y

    # Synthetic dataset
    n_normal = int(n_samples * 0.9)
    n_anom = n_samples - n_normal

    # Normal traffic: centered at 0
    Xn = np.random.normal(loc=0.0, scale=1.0, size=(n_normal, n_features))
    # Anomalous traffic: shifted mean + different covariance
    Xa = np.random.normal(loc=2.0, scale=1.2, size=(n_anom, n_features))

    X = np.vstack([Xn, Xa])
    y = np.hstack([np.zeros(n_normal, dtype=int), np.ones(n_anom, dtype=int)])

    # Shuffle
    p = np.random.permutation(len(y))
    return X[p], y[p]


def preprocess(X: np.ndarray, n_components: int = None) -> Tuple[np.ndarray, object]:
    """Scale (and optionally apply PCA) to reduce dimensionality for the quantum layer."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if n_components is not None and n_components < Xs.shape[1]:
        pca = PCA(n_components=n_components, random_state=SEED)
        Xr = pca.fit_transform(Xs)
        return Xr, (scaler, pca)
    else:
        return Xs, (scaler, None)

# PennyLane device
dev = qml.device("default.qubit", wires=QUANTUM_QUBITS)


def angle_embedding(x):
    """Simple angle embedding of features into rotations"""
    for i in range(QUANTUM_QUBITS):
        qml.RY(x[i], wires=i)


@qml.qnode(dev, interface='torch')
def quantum_circuit(inputs, weights):
    """Variational circuit: embed inputs, apply trainable rotations and entangling layers
    Returns expectation values of Z on each wire (can be reduced/pooled).
    - inputs: length QUANTUM_QUBITS
    - weights: shape (n_layers, QUANTUM_QUBITS, 3) if using RX/RY/RZ per wire
    """
    # inputs must be a torch tensor; convert to numpy if needed by pennylane
    # Angle embedding
    for i in range(QUANTUM_QUBITS):
        qml.RY(inputs[i], wires=i)

    n_layers = weights.shape[0]
    for l in range(n_layers):
        for w in range(QUANTUM_QUBITS):
            qml.RX(weights[l, w, 0], wires=w)
            qml.RY(weights[l, w, 1], wires=w)
            qml.RZ(weights[l, w, 2], wires=w)
        # entangling layer (ring)
        for w in range(QUANTUM_QUBITS):
            qml.CNOT(wires=[w, (w + 1) % QUANTUM_QUBITS])

    # Measurements: return expectation of PauliZ on each wire
    return [qml.expval(qml.PauliZ(i)) for i in range(QUANTUM_QUBITS)]


class QuantumLayer(nn.Module):
    def __init__(self, n_layers: int = 2):
        super().__init__()
        # Initialize weight tensor for the variational circuit
        # shape: (n_layers, n_qubits, 3)
        weight_shape = (n_layers, QUANTUM_QUBITS, 3)
        # Use a torch Parameter so optimizer updates it
        self.weights = nn.Parameter(torch.randn(weight_shape) * 0.1)
        self.n_layers = n_layers

    def forward(self, x):
        # x expected shape: (batch, QUANTUM_QUBITS)
        outputs = []
        for i in range(x.shape[0]):
            out = quantum_circuit(x[i], self.weights)
            outputs.append(out)
        return torch.stack(outputs)

class HybridNet(nn.Module):
    def __init__(self, input_dim: int, classical_latent: int = CLASSICAL_LATENT, quantum_layers: int = 2, n_classes: int = 2):
        super().__init__()
        # Classical encoder
        self.classical_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, classical_latent),
            nn.ReLU()
        )

        # Quantum layer will accept only a small subset of features (quantum-friendly dim)
        # We'll map part of the classical latent to the quantum input size
        self.map_to_quantum = nn.Linear(classical_latent, QUANTUM_INPUT_DIM)
        self.quantum = QuantumLayer(n_layers=quantum_layers)

        # Final head combines classical latent + quantum outputs
        combined_dim = classical_latent + QUANTUM_QUBITS
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        # x: (batch, input_dim)
        c = self.classical_encoder(x)
        q_in = torch.tanh(self.map_to_quantum(c))  # bound values for rotations
        q_out = self.quantum(q_in)
        # concatenate classical latent and quantum outputs
        combined = torch.cat([c, q_out], dim=1)
        logits = self.head(combined)
        return logits

def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE).float()
            yb = yb.to(DEVICE).long()
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE).float()
                yb = yb.to(DEVICE).long()
                out = model(xb)
                preds = out.argmax(dim=1).cpu().numpy()
                y_pred.extend(preds.tolist())
                y_true.extend(yb.cpu().numpy().tolist())

        acc = accuracy_score(y_true, y_pred)
        print(f"Epoch {epoch:02d} | Loss: {epoch_loss:.4f} | Val Acc: {acc:.4f}")

    return model


def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE).float()
            yb = yb.to(DEVICE).long()
            out = model(xb)
            preds = out.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(yb.cpu().numpy().tolist())

    print("Test Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))

def main(args):
    print("Loading data...")
    X, y = load_or_generate_data(args.data, n_features=args.features, n_samples=args.samples)
    Xp, pp = preprocess(X, n_components=args.pca_components)

    # If PCA reduced too much, make sure we still have N_FEATURES
    input_dim = Xp.shape[1]
    print(f"Dataset shapes: X={X.shape} -> after preprocess {Xp.shape}, labels={np.unique(y)}")

    X_train, X_temp, y_train, y_temp = train_test_split(Xp, y, test_size=0.3, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

    # Convert to torch tensors
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    model = HybridNet(input_dim=input_dim, classical_latent=args.classical_latent, quantum_layers=args.quantum_layers, n_classes=len(np.unique(y)))
    print(model)

    # Train
    model = train_model(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr)

    # Evaluate
    evaluate_model(model, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hybrid QAI-QML cybersecurity demo')
    parser.add_argument('--data', type=str, default=None, help='Path to CSV with features + label column (optional)')
    parser.add_argument('--features', type=int, default=N_FEATURES, help='Number of synthetic features to create if no data file')
    parser.add_argument('--samples', type=int, default=2000, help='Number of synthetic samples to generate')
    parser.add_argument('--pca-components', type=int, default=None, help='If set, apply PCA to reduce features to this many components')
    parser.add_argument('--classical-latent', type=int, default=CLASSICAL_LATENT)
    parser.add_argument('--quantum-layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LR)

    args = parser.parse_args()
    main(args)
