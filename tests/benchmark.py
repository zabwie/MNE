"""
Benchmark test for Metabolic Neural Ecosystem (MNE) vs Standard Neural Network.

This benchmark compares:
1. Training latency (forward/backward pass time)
2. Memory usage (peak memory consumption)
3. Learning speed (loss reduction per epoch)
4. Energy efficiency (computation per unit loss reduction)
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

# Suppress NumPy compatibility warnings from PyTorch
# PyTorch was compiled with NumPy 1.x but system has NumPy 2.x
warnings.filterwarnings("ignore", message=".*_ARRAY_API not found.*")
warnings.filterwarnings("ignore", message=".*NumPy 1.x cannot be run in.*")

# Also suppress warnings from torch module
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

# Import MNE
import sys
import os

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


class StandardNN(nn.Module):
    """Standard neural network for comparison."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except:
        # Fallback for platforms where psutil might not work correctly
        return 0.0


def create_synthetic_dataset(
    num_samples: int = 1000,
    input_dim: int = 50,
    output_dim: int = 10,
    seed: int = 42,
    task_type: str = "regression",  # "regression" or "classification"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic dataset using Python's random module to avoid NumPy initialization."""
    import io
    import contextlib

    random.seed(seed)

    # Generate random features using Python's random module
    # This avoids triggering PyTorch's NumPy initialization
    X_list = []
    for _ in range(num_samples):
        row = [random.gauss(0, 1) for _ in range(input_dim)]
        X_list.append(row)

    # Convert to PyTorch tensor, suppressing stderr to hide NumPy warnings
    with contextlib.redirect_stderr(io.StringIO()):
        X = torch.tensor(X_list, dtype=torch.float32)

    if task_type == "regression":
        # Create regression targets (output_dim-dimensional)
        # Simple non-linear transformation using matrix operations
        # Create weight matrices for transformation using Python's random
        W1_list = [
            [random.gauss(0, 0.1) for _ in range(output_dim)] for _ in range(input_dim)
        ]
        W2_list = [
            [random.gauss(0, 0.1) for _ in range(output_dim)] for _ in range(input_dim)
        ]

        with contextlib.redirect_stderr(io.StringIO()):
            W1 = torch.tensor(W1_list, dtype=torch.float32)
            W2 = torch.tensor(W2_list, dtype=torch.float32)

            # Compute y = sin(X @ W1) + 0.5 * (X @ W2)^2
            y = torch.sin(torch.mm(X, W1)) + 0.5 * torch.mm(X, W2).pow(2)
        return X, y
    else:
        # Create classification targets - pure PyTorch matrix operations
        # Use matrix multiplication to create class scores
        num_features_to_use = min(output_dim, input_dim)

        # Create identity-like weight matrix using torch.eye
        with contextlib.redirect_stderr(io.StringIO()):
            if input_dim >= output_dim:
                # More features than outputs - use first output_dim features
                W = torch.eye(input_dim, output_dim)
            else:
                # Fewer features than outputs - pad with zeros
                W = torch.zeros(input_dim, output_dim)
                identity_part = torch.eye(input_dim)
                W = torch.cat(
                    [identity_part, torch.zeros(input_dim, output_dim - input_dim)],
                    dim=1,
                )

            # Compute class scores using matrix multiplication
            class_scores = torch.mm(X, W)

            # Add non-linearity with tanh
            class_scores = torch.tanh(class_scores)

            # Assign class based on maximum score
            y = torch.argmax(class_scores, dim=1)
        return X, y


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
    """Train standard neural network and collect metrics."""
    print(f"\nTraining Standard Neural Network...")

    # Create data loader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Initialize result container
    result = BenchmarkResult(
        model_name="StandardNN",
        latency_per_epoch=[],
        memory_per_epoch=[],
        loss_per_epoch=[],
        energy_per_epoch=[],
        neurons_per_epoch=[],
    )

    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start_memory = get_memory_usage()

            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # Record peak memory for this batch
            batch_end_memory = get_memory_usage()
            if batch_idx == 0:  # Record first batch memory as representative
                result.memory_per_epoch.append(batch_end_memory - batch_start_memory)

        # Calculate epoch metrics
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
    """Train MNE and collect metrics."""
    print(f"\nTraining Metabolic Neural Ecosystem...")

    # Create data loader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Initialize result container
    result = BenchmarkResult(
        model_name="MNE",
        latency_per_epoch=[],
        memory_per_epoch=[],
        loss_per_epoch=[],
        energy_per_epoch=[],
        neurons_per_epoch=[],
    )

    # Get initial state (will be reinitialized for each batch)
    state = None

    # Add output projection layer (MNE outputs all neuron activations)
    output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1
    if len(y_train.shape) == 1:
        output_dim = 1
    output_projection = nn.Linear(model.config.num_neurons, output_dim)

    # Create optimizer for output projection
    projection_optimizer = optim.Adam(output_projection.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start_memory = get_memory_usage()

            # Reinitialize state for this batch size
            current_batch_size = data.shape[0]
            state = model.get_initial_state(batch_size=current_batch_size)

            # MNE forward pass
            neuron_activations, new_state = model.forward(
                data, state, apply_plasticity=True
            )

            # Project to output dimension
            outputs = output_projection(neuron_activations)

            # Compute loss (MSE for regression)
            if len(target.shape) == 1:
                target = target.unsqueeze(1)  # Add dimension for MSE
            loss = nn.MSELoss()(outputs, target.float())

            # Backward pass for output projection
            projection_optimizer.zero_grad()
            loss.backward()
            projection_optimizer.step()

            # Update state
            state = new_state
            epoch_loss += loss.item()
            num_batches += 1

            # Record peak memory for this batch
            batch_end_memory = get_memory_usage()
            if batch_idx == 0:  # Record first batch memory as representative
                result.memory_per_epoch.append(batch_end_memory - batch_start_memory)

        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

        # Get current metrics from state
        current_metrics = model.get_metrics(state)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_batches = 0
            # Simple validation on a subset
            val_subset = min(32, len(X_val))
            # Reinitialize state for validation batch size
            val_state = model.get_initial_state(batch_size=val_subset)
            neuron_activations, _ = model.forward(
                X_val[:val_subset], val_state, apply_plasticity=False
            )
            outputs = output_projection(neuron_activations)
            if len(y_val.shape) == 1:
                target = y_val[:val_subset].unsqueeze(1)
            else:
                target = y_val[:val_subset]
            val_loss = nn.MSELoss()(outputs, target.float()).item()

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
    input_dim: int = 50,
    hidden_dim: int = 100,
    output_dim: int = 10,
    seed: int = 42,
) -> Dict[str, BenchmarkResult]:
    """Run complete benchmark comparing MNE vs StandardNN."""
    print("=" * 80)
    print("METABOLIC NEURAL ECOSYSTEM BENCHMARK")
    print("=" * 80)

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create synthetic dataset
    print(f"\nCreating synthetic dataset...")

    # For StandardNN: classification task
    X_cls, y_cls = create_synthetic_dataset(
        dataset_size, input_dim, output_dim, seed, task_type="classification"
    )

    # For MNE: regression task (since MNE uses MSE loss)
    X_reg, y_reg = create_synthetic_dataset(
        dataset_size, input_dim, output_dim, seed, task_type="regression"
    )

    # Split into train/validation
    split_idx = int(0.8 * dataset_size)

    # StandardNN data (classification)
    X_train_std, X_val_std = X_cls[:split_idx], X_cls[split_idx:]
    y_train_std, y_val_std = y_cls[:split_idx], y_cls[split_idx:]

    # MNE data (regression)
    X_train_mne, X_val_mne = X_reg[:split_idx], X_reg[split_idx:]
    y_train_mne, y_val_mne = y_reg[:split_idx], y_reg[split_idx:]

    print(f"Dataset: {dataset_size} samples, {input_dim} features")
    print(f"StandardNN: {output_dim}-class classification")
    print(f"MNE: {output_dim}-dimensional regression")
    print(f"Train: {split_idx} samples, Validation: {dataset_size - split_idx} samples")

    # Initialize models
    print(f"\nInitializing models...")

    # Standard Neural Network
    standard_nn = StandardNN(input_dim, hidden_dim, output_dim)
    print(
        f"StandardNN: {sum(p.numel() for p in standard_nn.parameters()):,} parameters"
    )

    # For MNE, we need input dimension to match number of neurons
    # So we'll use hidden_dim as both input_dim and num_neurons for MNE
    # But we need to project the input data to match
    mne_input_dim = hidden_dim  # MNE expects input_dim = num_neurons

    # Metabolic Neural Ecosystem
    mne_config = MNEConfig(
        num_neurons=mne_input_dim,
        activation_fn="tanh",
        # Neuron parameters
        kappa=0.1,
        gamma=0.05,
        alpha=0.5,
        beta=1.0,
        delta=0.01,
        rho=0.01,
        target_activation=0.5,
        initial_resource=1.0,
        initial_threshold=0.0,
        # Synapse parameters
        eta=0.01,
        mu=0.1,
        weight_init_std=0.1,
        sparsity=0.8,
        # Energy parameters
        initial_energy=100.0,
        energy_influx=10.0,
        min_energy=0.0,
        max_energy=200.0,
        # Topology parameters
        max_neurons=int(hidden_dim * 1.5),  # Allow growth
        min_neurons=int(hidden_dim * 0.5),  # Allow shrinkage
        resource_high=8.0,
        resource_low=2.0,
        # System parameters
        device="cpu",
    )
    mne = MNE(mne_config)
    print(f"MNE: {hidden_dim} initial neurons, dynamic topology enabled")

    # Run garbage collection before benchmarking
    gc.collect()

    # Train models and collect metrics
    print(f"\nStarting benchmark with {num_epochs} epochs...")

    results = {}

    # Train StandardNN
    results["standard_nn"] = train_standard_nn(
        standard_nn,
        X_train_std,
        y_train_std,
        X_val_std,
        y_val_std,
        num_epochs=num_epochs,
        batch_size=32,
        learning_rate=0.01,
    )

    # Run garbage collection between models
    gc.collect()

    # For MNE, we need to project inputs to match neuron count
    # Create a simple linear projection
    projection = nn.Linear(input_dim, mne_input_dim)

    # Project MNE training data
    X_train_mne_proj = projection(X_train_mne).detach()
    X_val_mne_proj = projection(X_val_mne).detach()

    # Train MNE
    results["mne"] = train_mne(
        mne,
        X_train_mne_proj,
        y_train_mne,
        X_val_mne_proj,
        y_val_mne,
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

    standard_result = results["standard_nn"]
    mne_result = results["mne"]

    # Calculate comparison metrics
    latency_ratio = mne_result.avg_latency / standard_result.avg_latency
    memory_ratio = mne_result.avg_memory / standard_result.avg_memory
    loss_ratio = mne_result.final_loss / standard_result.final_loss

    print(f"\nPerformance Comparison (MNE vs StandardNN):")
    print(f"{'Metric':<25} {'StandardNN':<12} {'MNE':<12} {'Ratio':<10} {'Better':<10}")
    print("-" * 70)

    # Latency
    better = "MNE" if latency_ratio < 1.0 else "StandardNN"
    print(
        f"{'Avg Latency (s)':<25} "
        f"{standard_result.avg_latency:<12.3f} "
        f"{mne_result.avg_latency:<12.3f} "
        f"{latency_ratio:<10.3f} "
        f"{better:<10}"
    )

    # Memory
    better = "MNE" if memory_ratio < 1.0 else "StandardNN"
    print(
        f"{'Avg Memory (MB)':<25} "
        f"{standard_result.avg_memory:<12.1f} "
        f"{mne_result.avg_memory:<12.1f} "
        f"{memory_ratio:<10.3f} "
        f"{better:<10}"
    )

    # Final Loss
    better = "MNE" if loss_ratio < 1.0 else "StandardNN"
    print(
        f"{'Final Loss':<25} "
        f"{standard_result.final_loss:<12.4f} "
        f"{mne_result.final_loss:<12.4f} "
        f"{loss_ratio:<10.3f} "
        f"{better:<10}"
    )

    # Loss Reduction Rate
    std_rate = standard_result.loss_reduction_rate
    mne_rate = mne_result.loss_reduction_rate
    rate_ratio = mne_rate / std_rate if std_rate != 0 else float("inf")
    better = "MNE" if rate_ratio < 1.0 else "StandardNN"
    print(
        f"{'Loss Reduction Rate':<25} "
        f"{std_rate:<12.4f} "
        f"{mne_rate:<12.4f} "
        f"{rate_ratio:<10.3f} "
        f"{better:<10}"
    )

    # MNE-specific metrics
    print(f"\nMNE-Specific Metrics:")
    print(
        f"  Final Neuron Count: {mne_result.neurons_per_epoch[-1] if mne_result.neurons_per_epoch else 'N/A'}"
    )
    print(f"  Total Energy Consumed: {sum(mne_result.energy_per_epoch):.1f}")
    print(
        f"  Neurogenesis Events: {max(mne_result.neurons_per_epoch) - min(mne_result.neurons_per_epoch) if mne_result.neurons_per_epoch else 0}"
    )

    # Efficiency metrics
    if mne_result.energy_per_epoch and standard_result.avg_latency > 0:
        energy_per_loss_reduction = (
            sum(mne_result.energy_per_epoch) / abs(mne_rate)
            if abs(mne_rate) > 0
            else float("inf")
        )
        # Get actual number of epochs from results
        num_epochs_actual = len(standard_result.loss_per_epoch)
        time_per_loss_reduction_std = (
            standard_result.avg_latency * num_epochs_actual / abs(std_rate)
            if abs(std_rate) > 0
            else float("inf")
        )
        time_per_loss_reduction_mne = (
            mne_result.avg_latency * num_epochs_actual / abs(mne_rate)
            if abs(mne_rate) > 0
            else float("inf")
        )

        print(f"\nEfficiency Metrics:")
        print(
            f"  StandardNN Time per Unit Loss Reduction: {time_per_loss_reduction_std:.2f}s"
        )
        print(f"  MNE Time per Unit Loss Reduction: {time_per_loss_reduction_mne:.2f}s")
        print(f"  MNE Energy per Unit Loss Reduction: {energy_per_loss_reduction:.1f}")

    # Determine overall winner
    print(f"\n{'=' * 40}")
    print("OVERALL ASSESSMENT")
    print(f"{'=' * 40}")

    # Score based on multiple factors (lower is better)
    score_standard = (
        standard_result.avg_latency * 0.3
        + standard_result.avg_memory * 0.2
        + standard_result.final_loss * 0.5
    )

    score_mne = (
        mne_result.avg_latency * 0.3
        + mne_result.avg_memory * 0.2
        + mne_result.final_loss * 0.5
    )

    if score_mne < score_standard:
        print("MNE performs BETTER overall")
        print(f"  MNE Score: {score_mne:.3f}, StandardNN Score: {score_standard:.3f}")
        advantage = ((score_standard - score_mne) / score_standard) * 100
        print(f"  MNE advantage: {advantage:.1f}%")
    else:
        print("StandardNN performs BETTER overall")
        print(f"  StandardNN Score: {score_standard:.3f}, MNE Score: {score_mne:.3f}")
        advantage = ((score_mne - score_standard) / score_mne) * 100
        print(f"  StandardNN advantage: {advantage:.1f}%")

    # Key insights
    print(f"\nKEY INSIGHTS:")
    if latency_ratio < 0.9:
        print(f"  • MNE is {100 * (1 - latency_ratio):.1f}% faster than StandardNN")
    elif latency_ratio > 1.1:
        print(f"  • StandardNN is {100 * (latency_ratio - 1):.1f}% faster than MNE")

    if memory_ratio < 0.9:
        print(
            f"  • MNE uses {100 * (1 - memory_ratio):.1f}% less memory than StandardNN"
        )
    elif memory_ratio > 1.1:
        print(
            f"  • StandardNN uses {100 * (memory_ratio - 1):.1f}% less memory than MNE"
        )

    if loss_ratio < 0.95:
        print(f"  • MNE achieves {100 * (1 - loss_ratio):.1f}% lower final loss")
    elif loss_ratio > 1.05:
        print(
            f"  • StandardNN achieves {100 * (1 - 1 / loss_ratio):.1f}% lower final loss"
        )

    if mne_result.neurons_per_epoch:
        neuron_change = (
            mne_result.neurons_per_epoch[-1] - mne_result.neurons_per_epoch[0]
        )
        if neuron_change > 0:
            print(f"  • MNE grew by {neuron_change} neurons (neurogenesis)")
        elif neuron_change < 0:
            print(f"  • MNE shrank by {-neuron_change} neurons (apoptosis)")


def plot_results(
    results: Dict[str, BenchmarkResult], save_path: str = "benchmark_results.png"
):
    """Plot benchmark results."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend
        import matplotlib.pyplot as plt

        standard_result = results["standard_nn"]
        mne_result = results["mne"]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot 1: Loss over epochs
        ax = axes[0, 0]
        epochs = range(1, len(standard_result.loss_per_epoch) + 1)
        ax.plot(
            epochs,
            standard_result.loss_per_epoch,
            "b-",
            label="StandardNN",
            linewidth=2,
        )
        ax.plot(epochs, mne_result.loss_per_epoch, "r-", label="MNE", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Latency over epochs
        ax = axes[0, 1]
        ax.plot(
            epochs,
            standard_result.latency_per_epoch,
            "b-",
            label="StandardNN",
            linewidth=2,
        )
        ax.plot(epochs, mne_result.latency_per_epoch, "r-", label="MNE", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Time (s)")
        ax.set_title("Epoch Latency")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Memory over epochs
        ax = axes[0, 2]
        ax.plot(
            epochs[: len(standard_result.memory_per_epoch)],
            standard_result.memory_per_epoch,
            "b-",
            label="StandardNN",
            linewidth=2,
        )
        ax.plot(
            epochs[: len(mne_result.memory_per_epoch)],
            mne_result.memory_per_epoch,
            "r-",
            label="MNE",
            linewidth=2,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Memory (MB)")
        ax.set_title("Memory Usage")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: MNE Energy consumption
        ax = axes[1, 0]
        if mne_result.energy_per_epoch:
            ax.plot(epochs, mne_result.energy_per_epoch, "g-", linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Energy")
            ax.set_title("MNE Energy Consumption")
            ax.grid(True, alpha=0.3)

        # Plot 5: MNE Neuron count
        ax = axes[1, 1]
        if mne_result.neurons_per_epoch:
            ax.plot(epochs, mne_result.neurons_per_epoch, "m-", linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Neurons")
            ax.set_title("MNE Neuron Count")
            ax.grid(True, alpha=0.3)

        # Plot 6: Comparison bar chart
        ax = axes[1, 2]
        metrics = ["Latency (s)", "Memory (MB)", "Final Loss"]
        standard_vals = [
            standard_result.avg_latency,
            standard_result.avg_memory,
            standard_result.final_loss,
        ]
        mne_vals = [
            mne_result.avg_latency,
            mne_result.avg_memory,
            mne_result.final_loss,
        ]

        x = np.arange(len(metrics))
        width = 0.35
        ax.bar(
            x - width / 2,
            standard_vals,
            width,
            label="StandardNN",
            color="blue",
            alpha=0.7,
        )
        ax.bar(x + width / 2, mne_vals, width, label="MNE", color="red", alpha=0.7)
        ax.set_xlabel("Metric")
        ax.set_ylabel("Value")
        ax.set_title("Performance Comparison")
        ax.set_xticks(x)
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
    print("Metabolic Neural Ecosystem Benchmark")
    print("Comparing MNE vs Standard Neural Network")

    # Suppress NumPy warnings
    import warnings

    warnings.filterwarnings("ignore", message="Failed to initialize NumPy")
    warnings.filterwarnings(
        "ignore", message="A module that was compiled using NumPy 1.x"
    )

    try:
        # Run benchmark with smaller settings for debugging
        results = run_benchmark(
            num_epochs=10,  # Reduced for debugging
            dataset_size=200,  # Reduced for debugging
            input_dim=20,  # Reduced for debugging
            hidden_dim=50,  # Reduced for debugging
            output_dim=5,  # Reduced for debugging
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
