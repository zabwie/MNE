"""
Comprehensive benchmark comparing RNN, LSTM, StandardNN, and MNE.

This benchmark compares four neural network architectures on identical sequence prediction tasks:
1. RNN (Recurrent Neural Network)
2. LSTM (Long Short-Term Memory)
3. StandardNN (Standard Feedforward Neural Network)
4. MNE (Metabolic Neural Ecosystem)

Metrics tracked:
- Training latency (forward/backward pass time)
- Memory usage (peak memory consumption)
- Learning speed (loss reduction per epoch)
- Energy efficiency (computation per unit loss reduction)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import psutil
import os
import gc
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import warnings
import sys
import random
import io
import contextlib

# Suppress NumPy compatibility warnings from PyTorch
warnings.filterwarnings("ignore", message=".*_ARRAY_API not found.*")
warnings.filterwarnings("ignore", message=".*NumPy 1.x cannot be run in.*")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

# Import MNE
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core import MNE, MNEConfig


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    model_name: str
    latency_per_epoch: List[float]  # seconds
    memory_per_epoch: List[float]  # MB
    loss_per_epoch: List[float]  # loss values
    energy_per_epoch: List[float]  # energy consumption (MNE only)
    neurons_per_epoch: List[int]  # number of neurons (MNE only)
    parameters: int  # total number of parameters

    @property
    def avg_latency(self) -> float:
        return np.mean(self.latency_per_epoch) if self.latency_per_epoch else 0.0

    @property
    def avg_memory(self) -> float:
        return np.mean(self.memory_per_epoch) if self.memory_per_epoch else 0.0

    @property
    def final_loss(self) -> float:
        return self.loss_per_epoch[-1] if self.loss_per_epoch else float("inf")

    @property
    def loss_reduction_rate(self) -> float:
        """Rate of loss reduction per epoch (negative is good)."""
        if len(self.loss_per_epoch) < 2:
            return 0.0
        return (self.loss_per_epoch[-1] - self.loss_per_epoch[0]) / len(
            self.loss_per_epoch
        )


class RNNModel(nn.Module):
    """Simple RNN model for sequence prediction."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.rnn(x)
        # Use the last time step's output
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


class LSTMModel(nn.Module):
    """LSTM model for sequence prediction."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.lstm(x)
        # Use the last time step's output
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


class StandardNN(nn.Module):
    """Standard feedforward neural network for sequence prediction."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # Flatten sequence for feedforward network
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.shape[0]
        # Flatten sequence
        x = x.view(batch_size, -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class RecurrentMNE(nn.Module):
    """
    Recurrent wrapper for MNE to process full sequences step-by-step.

    This wrapper makes MNE comparable to RNN/LSTM by processing entire sequences
    and maintaining state across time steps, rather than only seeing the last step.
    """

    def __init__(self, mne: MNE, input_dim: int, output_dim: int):
        super().__init__()
        self.mne = mne
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Input projection: project input_dim to num_neurons
        self.input_projection = nn.Linear(input_dim, mne.config.num_neurons)

        # Output projection: project num_neurons to output_dim
        self.output_projection = nn.Linear(mne.config.num_neurons, output_dim)

        # Store last neuron activations for contribution computation
        self.last_neuron_activations = None

    def forward(
        self,
        x: torch.Tensor,
        state=None,
        apply_plasticity: bool = False,
        contribution=None,
        return_activations: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Process sequence step-by-step.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            state: Optional MNE state to continue from
            apply_plasticity: Whether to apply plasticity during forward pass
            contribution: Optional contribution signal for plasticity
            return_activations: Whether to return neuron activations

        Returns:
            outputs: Output tensor of shape (batch_size, output_dim)
            new_state: Updated MNE state
            (optional) neuron_activations: Last neuron activations
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Initialize state if not provided
        if state is None:
            state = self.mne.get_initial_state(batch_size=batch_size)

        # Process sequence step-by-step
        for t in range(seq_len):
            # Get current time step
            current_step = x[:, t, :]  # (batch_size, input_dim)

            # Project input to match neuron count
            projected_input = self.input_projection(current_step)

            # Forward pass through MNE
            neuron_activations, state = self.mne.forward(
                projected_input,
                state,
                contribution=contribution,
                apply_plasticity=apply_plasticity,
            )

        # Store last neuron activations for contribution computation
        self.last_neuron_activations = neuron_activations.detach()

        # After processing all time steps, project to output dimension
        outputs = self.output_projection(neuron_activations)

        if return_activations:
            return outputs, state, neuron_activations
        return outputs, state

    def compute_contribution(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient-based contribution |∂L/∂a_i|.

        Args:
            outputs: Network outputs
            targets: Target values

        Returns:
            Contribution of shape (batch_size, num_neurons)
        """
        # Compute loss
        loss = nn.MSELoss()(outputs, targets)

        # Get last neuron activations and enable gradient tracking
        if self.last_neuron_activations is None:
            # Fallback: use uniform contribution
            batch_size = outputs.shape[0]
            return torch.ones(
                batch_size, self.mne.config.num_neurons, device=outputs.device
            )

        neuron_activations = self.last_neuron_activations.clone()
        neuron_activations.requires_grad_(True)

        # Recompute outputs with gradient tracking
        outputs_with_grad = self.output_projection(neuron_activations)
        loss_recomputed = nn.MSELoss()(outputs_with_grad, targets)

        # Compute gradient w.r.t. neuron activations
        grad = torch.autograd.grad(
            loss_recomputed,
            neuron_activations,
            create_graph=False,
            retain_graph=False,
        )[0]

        # Contribution is absolute gradient
        contribution = torch.abs(grad)

        return contribution


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except:
        return 0.0


def create_sequence_dataset(
    num_samples: int = 1000,
    seq_len: int = 10,
    input_dim: int = 5,
    output_dim: int = 3,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic sequence prediction dataset.

    Generates sequences where the target is a function of the sequence history.
    This is ideal for testing RNN/LSTM capabilities.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # Generate random sequences
    X_list = []
    y_list = []

    for _ in range(num_samples):
        # Generate sequence
        seq = []
        for _ in range(seq_len):
            row = [random.gauss(0, 1) for _ in range(input_dim)]
            seq.append(row)

        # Target is a function of the sequence (e.g., sum of last 3 time steps)
        # This creates temporal dependencies
        with contextlib.redirect_stderr(io.StringIO()):
            seq_tensor = torch.tensor(seq, dtype=torch.float32)

            # Compute target as weighted sum of sequence elements
            # More recent time steps have higher weights
            weights = torch.tensor([i + 1 for i in range(seq_len)], dtype=torch.float32)
            weights = weights / weights.sum()  # Normalize

            # Weighted sum across sequence
            weighted_sum = torch.sum(seq_tensor * weights.view(-1, 1), dim=0)

            # Apply non-linear transformation
            target = torch.sin(weighted_sum[:output_dim]) + 0.5 * torch.cos(
                weighted_sum[:output_dim]
            )

            X_list.append(seq_tensor)
            y_list.append(target)

    with contextlib.redirect_stderr(io.StringIO()):
        X = torch.stack(X_list)
        y = torch.stack(y_list)

    return X, y


def count_parameters(model: nn.Module) -> int:
    """Count total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_rnn(
    model: RNNModel,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.01,
) -> BenchmarkResult:
    """Train RNN model and collect metrics."""
    print(f"\nTraining RNN...")

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    result = BenchmarkResult(
        model_name="RNN",
        latency_per_epoch=[],
        memory_per_epoch=[],
        loss_per_epoch=[],
        energy_per_epoch=[],
        neurons_per_epoch=[],
        parameters=count_parameters(model),
    )

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start_memory = get_memory_usage()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            batch_end_memory = get_memory_usage()
            if batch_idx == 0:
                result.memory_per_epoch.append(batch_end_memory - batch_start_memory)

        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / num_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val).item()

        result.latency_per_epoch.append(epoch_time)
        result.loss_per_epoch.append(val_loss)

        print(
            f"  Epoch {epoch + 1}/{num_epochs}: "
            f"Loss={val_loss:.4f}, Time={epoch_time:.2f}s, "
            f"Memory={result.memory_per_epoch[-1]:.1f}MB"
        )

    return result


def train_lstm(
    model: LSTMModel,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.01,
) -> BenchmarkResult:
    """Train LSTM model and collect metrics."""
    print(f"\nTraining LSTM...")

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    result = BenchmarkResult(
        model_name="LSTM",
        latency_per_epoch=[],
        memory_per_epoch=[],
        loss_per_epoch=[],
        energy_per_epoch=[],
        neurons_per_epoch=[],
        parameters=count_parameters(model),
    )

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start_memory = get_memory_usage()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            batch_end_memory = get_memory_usage()
            if batch_idx == 0:
                result.memory_per_epoch.append(batch_end_memory - batch_start_memory)

        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / num_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val).item()

        result.latency_per_epoch.append(epoch_time)
        result.loss_per_epoch.append(val_loss)

        print(
            f"  Epoch {epoch + 1}/{num_epochs}: "
            f"Loss={val_loss:.4f}, Time={epoch_time:.2f}s, "
            f"Memory={result.memory_per_epoch[-1]:.1f}MB"
        )

    return result


def train_standard_nn(
    model: StandardNN,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.01,
) -> BenchmarkResult:
    """Train StandardNN model and collect metrics."""
    print(f"\nTraining StandardNN...")

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    result = BenchmarkResult(
        model_name="StandardNN",
        latency_per_epoch=[],
        memory_per_epoch=[],
        loss_per_epoch=[],
        energy_per_epoch=[],
        neurons_per_epoch=[],
        parameters=count_parameters(model),
    )

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start_memory = get_memory_usage()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            batch_end_memory = get_memory_usage()
            if batch_idx == 0:
                result.memory_per_epoch.append(batch_end_memory - batch_start_memory)

        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / num_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val).item()

        result.latency_per_epoch.append(epoch_time)
        result.loss_per_epoch.append(val_loss)

        print(
            f"  Epoch {epoch + 1}/{num_epochs}: "
            f"Loss={val_loss:.4f}, Time={epoch_time:.2f}s, "
            f"Memory={result.memory_per_epoch[-1]:.1f}MB"
        )

    return result


def train_mne(
    model: MNE,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.01,
) -> BenchmarkResult:
    """Train MNE model with recurrent wrapper for fair sequence processing."""
    print(f"\nTraining MNE (with recurrent sequence processing)...")

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Create recurrent wrapper for MNE to process full sequences
    input_dim = X_train.shape[2]
    output_dim = y_train.shape[1]
    recurrent_mne = RecurrentMNE(model, input_dim, output_dim)

    # Count ALL parameters: MNE core + input/output projections
    total_params = count_parameters(recurrent_mne)

    result = BenchmarkResult(
        model_name="MNE",
        latency_per_epoch=[],
        memory_per_epoch=[],
        loss_per_epoch=[],
        energy_per_epoch=[],
        neurons_per_epoch=[],
        parameters=total_params,
    )

    # Create optimizer for all parameters (MNE + projections)
    optimizer = optim.Adam(recurrent_mne.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start_memory = get_memory_usage()

            # Process full sequence step-by-step using RecurrentMNE
            # data shape: (batch_size, seq_len, input_dim)

            # First forward pass without plasticity to compute contribution
            outputs, state = recurrent_mne.forward(
                data, state=None, apply_plasticity=False
            )

            # Compute loss
            loss = nn.MSELoss()(outputs, target)

            # Compute contribution signal (gradient-based w.r.t. neuron activations)
            contribution = recurrent_mne.compute_contribution(outputs, target)

            # Second forward pass with plasticity using contribution
            outputs, new_state = recurrent_mne.forward(
                data, state=None, contribution=contribution, apply_plasticity=True
            )

            # Backward pass for all parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            batch_end_memory = get_memory_usage()
            if batch_idx == 0:
                result.memory_per_epoch.append(batch_end_memory - batch_start_memory)

        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

        # Get current metrics from state
        current_metrics = model.get_metrics(new_state)

        # Validation - process full sequences
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_batches = 0
            val_subset = min(32, len(X_val))
            val_outputs, _ = recurrent_mne.forward(
                X_val[:val_subset], state=None, apply_plasticity=False
            )
            val_loss = nn.MSELoss()(val_outputs, y_val[:val_subset]).item()

        result.latency_per_epoch.append(epoch_time)
        result.loss_per_epoch.append(val_loss)
        result.energy_per_epoch.append(current_metrics.get("total_energy", 0.0))
        result.neurons_per_epoch.append(current_metrics.get("num_neurons", 0))

        print(
            f"  Epoch {epoch + 1}/{num_epochs}: "
            f"Loss={val_loss:.4f}, Time={epoch_time:.2f}s, "
            f"Memory={result.memory_per_epoch[-1]:.1f}MB, "
            f"Neurons={current_metrics.get('num_neurons', 0)}, "
            f"Energy={current_metrics.get('total_energy', 0.0):.1f}"
        )

    return result


def run_benchmark(
    num_epochs: int = 20,
    dataset_size: int = 1000,
    seq_len: int = 10,
    input_dim: int = 5,
    hidden_dim: int = 64,
    output_dim: int = 3,
    seed: int = 42,
) -> Dict[str, BenchmarkResult]:
    """Run complete benchmark comparing RNN, LSTM, StandardNN, and MNE."""
    print("=" * 80)
    print("COMPREHENSIVE NEURAL NETWORK BENCHMARK")
    print("Comparing RNN, LSTM, StandardNN, and MNE")
    print("=" * 80)

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create sequence dataset
    print(f"\nCreating sequence dataset...")
    X, y = create_sequence_dataset(dataset_size, seq_len, input_dim, output_dim, seed)

    # Split into train/validation
    split_idx = int(0.8 * dataset_size)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(
        f"Dataset: {dataset_size} samples, {seq_len} time steps, {input_dim} features"
    )
    print(f"Output: {output_dim} dimensions")
    print(f"Train: {split_idx} samples, Validation: {dataset_size - split_idx} samples")

    # Initialize models
    print(f"\nInitializing models...")

    # RNN
    rnn = RNNModel(input_dim, hidden_dim, output_dim, num_layers=2)
    print(f"RNN: {count_parameters(rnn):,} parameters")

    # LSTM
    lstm = LSTMModel(input_dim, hidden_dim, output_dim, num_layers=2)
    print(f"LSTM: {count_parameters(lstm):,} parameters")

    # StandardNN (flattened input)
    standard_nn = StandardNN(seq_len * input_dim, hidden_dim, output_dim)
    print(f"StandardNN: {count_parameters(standard_nn):,} parameters")

    # MNE
    mne_config = MNEConfig(
        num_neurons=hidden_dim,
        activation_fn="tanh",
        kappa=0.1,
        gamma=0.05,
        alpha=0.5,
        beta=1.0,
        delta=0.01,
        rho=0.01,
        target_activation=0.5,
        initial_resource=1.0,
        initial_threshold=0.0,
        eta=0.01,
        mu=0.1,
        weight_init_std=0.1,
        sparsity=0.8,
        initial_energy=100.0,
        energy_influx=10.0,
        min_energy=20.0,
        max_energy=200.0,
        max_neurons=int(hidden_dim * 1.5),
        min_neurons=int(hidden_dim * 0.5),
        resource_high=2.0,
        resource_low=0.1,
        device="cpu",
    )
    mne = MNE(mne_config)
    print(f"MNE: {hidden_dim} initial neurons, dynamic topology enabled")

    # Run garbage collection before benchmarking
    gc.collect()

    # Train models and collect metrics
    print(f"\nStarting benchmark with {num_epochs} epochs...")

    results = {}

    # Train RNN
    results["rnn"] = train_rnn(
        rnn,
        X_train,
        y_train,
        X_val,
        y_val,
        num_epochs=num_epochs,
        batch_size=32,
        learning_rate=0.01,
    )
    gc.collect()

    # Train LSTM
    results["lstm"] = train_lstm(
        lstm,
        X_train,
        y_train,
        X_val,
        y_val,
        num_epochs=num_epochs,
        batch_size=32,
        learning_rate=0.01,
    )
    gc.collect()

    # Train StandardNN
    results["standard_nn"] = train_standard_nn(
        standard_nn,
        X_train,
        y_train,
        X_val,
        y_val,
        num_epochs=num_epochs,
        batch_size=32,
        learning_rate=0.01,
    )
    gc.collect()

    # Train MNE
    results["mne"] = train_mne(
        mne,
        X_train,
        y_train,
        X_val,
        y_val,
        num_epochs=num_epochs,
        batch_size=32,
        learning_rate=0.01,
    )

    return results


def analyze_results(results: Dict[str, BenchmarkResult]):
    """Analyze and compare benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK ANALYSIS")
    print("=" * 80)

    # Print parameter counts
    print(f"\nModel Parameters:")
    for name, result in results.items():
        print(f"  {name}: {result.parameters:,} parameters")

    # Performance comparison table
    print(f"\nPerformance Comparison:")
    print(
        f"{'Model':<15} {'Params':<12} {'Latency(s)':<12} {'Memory(MB)':<12} {'Final Loss':<12} {'Loss Rate':<12}"
    )
    print("-" * 80)

    for name, result in results.items():
        print(
            f"{name:<15} "
            f"{result.parameters:<12,} "
            f"{result.avg_latency:<12.3f} "
            f"{result.avg_memory:<12.1f} "
            f"{result.final_loss:<12.4f} "
            f"{result.loss_reduction_rate:<12.4f}"
        )

    # Find best performers
    print(f"\nBest Performers:")

    # Best latency
    best_latency = min(results.items(), key=lambda x: x[1].avg_latency)
    print(
        f"  Fastest Training: {best_latency[0]} ({best_latency[1].avg_latency:.3f}s/epoch)"
    )

    # Best memory
    best_memory = min(results.items(), key=lambda x: x[1].avg_memory)
    print(f"  Lowest Memory: {best_memory[0]} ({best_memory[1].avg_memory:.1f}MB)")

    # Best loss
    best_loss = min(results.items(), key=lambda x: x[1].final_loss)
    print(f"  Best Final Loss: {best_loss[0]} ({best_loss[1].final_loss:.4f})")

    # Best loss reduction rate
    best_rate = min(results.items(), key=lambda x: x[1].loss_reduction_rate)
    print(
        f"  Fastest Learning: {best_rate[0]} ({best_rate[1].loss_reduction_rate:.4f}/epoch)"
    )

    # MNE-specific metrics
    if "mne" in results:
        mne_result = results["mne"]
        print(f"\nMNE-Specific Metrics:")
        print(
            f"  Final Neuron Count: {mne_result.neurons_per_epoch[-1] if mne_result.neurons_per_epoch else 'N/A'}"
        )
        print(f"  Total Energy Consumed: {sum(mne_result.energy_per_epoch):.1f}")
        if mne_result.neurons_per_epoch:
            neuron_change = (
                mne_result.neurons_per_epoch[-1] - mne_result.neurons_per_epoch[0]
            )
            if neuron_change > 0:
                print(f"  Neurogenesis: +{neuron_change} neurons")
            elif neuron_change < 0:
                print(f"  Apoptosis: {-neuron_change} neurons")

    # Efficiency analysis
    print(f"\nEfficiency Analysis:")
    for name, result in results.items():
        if result.avg_latency > 0 and abs(result.loss_reduction_rate) > 0:
            time_per_loss = result.avg_latency / abs(result.loss_reduction_rate)
            print(f"  {name}: {time_per_loss:.2f}s per unit loss reduction")

    # Overall ranking
    print(f"\nOverall Ranking (lower is better):")
    scores = {}
    for name, result in results.items():
        # Normalize metrics and compute composite score
        # Lower latency, memory, and loss are better
        score = (
            result.avg_latency * 0.3 + result.avg_memory * 0.2 + result.final_loss * 0.5
        )
        scores[name] = score

    # Sort by score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    for rank, (name, score) in enumerate(sorted_scores, 1):
        print(f"  {rank}. {name}: {score:.3f}")


def plot_results(
    results: Dict[str, BenchmarkResult], save_path: str = "final_comparison_results.png"
):
    """Plot benchmark results."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        colors = {"rnn": "blue", "lstm": "green", "standard_nn": "orange", "mne": "red"}

        # Plot 1: Loss over epochs
        ax = axes[0, 0]
        for name, result in results.items():
            epochs = range(1, len(result.loss_per_epoch) + 1)
            ax.plot(
                epochs,
                result.loss_per_epoch,
                label=name,
                color=colors.get(name, "black"),
                linewidth=2,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Latency over epochs
        ax = axes[0, 1]
        for name, result in results.items():
            epochs = range(1, len(result.latency_per_epoch) + 1)
            ax.plot(
                epochs,
                result.latency_per_epoch,
                label=name,
                color=colors.get(name, "black"),
                linewidth=2,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Time (s)")
        ax.set_title("Epoch Latency")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Memory over epochs
        ax = axes[0, 2]
        for name, result in results.items():
            epochs = range(1, len(result.memory_per_epoch) + 1)
            ax.plot(
                epochs,
                result.memory_per_epoch,
                label=name,
                color=colors.get(name, "black"),
                linewidth=2,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Memory (MB)")
        ax.set_title("Memory Usage")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: MNE Energy consumption
        ax = axes[1, 0]
        if "mne" in results and results["mne"].energy_per_epoch:
            epochs = range(1, len(results["mne"].energy_per_epoch) + 1)
            ax.plot(
                epochs,
                results["mne"].energy_per_epoch,
                color=colors["mne"],
                linewidth=2,
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Energy")
            ax.set_title("MNE Energy Consumption")
            ax.grid(True, alpha=0.3)

        # Plot 5: MNE Neuron count
        ax = axes[1, 1]
        if "mne" in results and results["mne"].neurons_per_epoch:
            epochs = range(1, len(results["mne"].neurons_per_epoch) + 1)
            ax.plot(
                epochs,
                results["mne"].neurons_per_epoch,
                color=colors["mne"],
                linewidth=2,
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Neurons")
            ax.set_title("MNE Neuron Count")
            ax.grid(True, alpha=0.3)

        # Plot 6: Comparison bar chart
        ax = axes[1, 2]
        metrics = ["Latency (s)", "Memory (MB)", "Final Loss"]
        x = np.arange(len(metrics))
        width = 0.2

        for i, (name, result) in enumerate(results.items()):
            values = [result.avg_latency, result.avg_memory, result.final_loss]
            ax.bar(
                x + i * width,
                values,
                width,
                label=name,
                color=colors.get(name, "black"),
                alpha=0.7,
            )

        ax.set_xlabel("Metric")
        ax.set_ylabel("Value")
        ax.set_title("Performance Comparison")
        ax.set_xticks(x + width * (len(results) - 1) / 2)
        ax.set_xticklabels(metrics, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nResults plotted and saved to {save_path}")

    except ImportError:
        print("\nMatplotlib not available. Skipping plots.")
    except Exception as e:
        print(f"\nCould not generate plots: {e}")


def main():
    """Main benchmark function."""
    print("Comprehensive Neural Network Benchmark")
    print("Comparing RNN, LSTM, StandardNN, and MNE")

    try:
        # Run benchmark
        results = run_benchmark(
            num_epochs=20,
            dataset_size=1000,
            seq_len=10,
            input_dim=5,
            hidden_dim=64,
            output_dim=3,
            seed=42,
        )

        # Analyze results
        analyze_results(results)

        # Plot results
        print("\nGenerating plots...")
        plot_results(results)
        print("Plots generated successfully!")

        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETE")
        print("=" * 80)

    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
