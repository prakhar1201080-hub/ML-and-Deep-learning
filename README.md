# Super QAI-QML Cybersecurity Hybrid

This project demonstrates a **hybrid classical-quantum AI (QAI-QML) pipeline** for cybersecurity anomaly detection. It combines a classical neural network with a variational quantum circuit to create a cutting-edge proof-of-concept model for detecting anomalous network activity.

---

## Features

* Classical encoder (MLP) extracts latent features from input data.
* Quantum variational layer (PennyLane) processes a subset of features.
* Hybrid concatenation of classical and quantum outputs for final classification.
* Supports synthetic datasets or custom CSVs with `label` column.
* Easily adjustable hyperparameters for experiments.

---

## Requirements

* Python 3.8+
* torch
* pennylane
* pennylane-lightning (optional, faster simulation)
* scikit-learn
* pandas, numpy

Install with:

```bash
pip install torch pennylane pennylane-lightning scikit-learn pandas numpy
```

---

## Usage

1. **Run with synthetic dataset (default):**

```bash
python super_qai_qml_cybersec.py
```

2. **Run with your own CSV:**

   * CSV must include a `label` column.

```bash
python super_qai_qml_cybersec.py --data path/to/your_dataset.csv
```

3. **Optional arguments:**

```bash
--features N          # Number of synthetic features to generate
--samples N           # Number of synthetic samples to generate
--pca-components N    # Apply PCA to reduce features
--classical-latent N  # Size of classical latent vector
--quantum-layers N    # Number of quantum variational layers
--epochs N            # Number of training epochs
--lr FLOAT            # Learning rate
```

---

## Architecture

```
Input Features -> Classical MLP Encoder -> Classical Latent Vector
                                         |
                                         v
                                Quantum Layer (PennyLane)
                                         |
                                         v
       Concatenate Classical Latent + Quantum Output -> Final MLP -> Class Prediction
```

* **Classical Encoder:** Linear layers + ReLU, reduces input to latent vector.
* **Quantum Layer:** Angle embedding + RX/RY/RZ rotations + entangling layers, outputs expectation values.
* **Final Head:** Combines classical latent + quantum outputs for prediction.

---

## Notes

* This is **proof-of-concept** for research/demo purposes, not production-grade.
* Designed for experimentation with quantum circuits in PyTorch.
* Adjust `QUANTUM_QUBITS` and latent size carefully for performance on simulators.

---

## References

* [PennyLane](https://pennylane.ai/)
* [PyTorch](https://pytorch.org/)
* Quantum machine learning concepts for cybersecurity anomaly detection.

---

## License

MIT License
