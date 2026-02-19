"""
Fair Comparison Benchmark for Neural Network Architectures.

This benchmark compares architectures with MATCHED COMPUTATIONAL BUDGETS,
not just parameter counts. The key principles are:

1. FLOPs Matching: All architectures have similar FLOPs per inference
2. Efficiency Metrics: Compare accuracy per joule, accuracy per parameter
3. Task Appropriateness: Use tasks where each architecture can shine
4. Energy-Constrained Scenarios: Test MNE's adaptive resource allocation

Architectures Compared:
- StandardNN: Feedforward neural network (baseline)
- MNE: Metabolic Neural Ecosystem (energy-efficient, adaptive)

Key Metrics:
- FLOPs per inference (matched across models)
- Accuracy per parameter (parameter efficiency)
- Accuracy per joule (energy efficiency)
- Dynamic resource allocation (MNE-specific)
- Adaptability under energy constraints

Reference:
- Levy, W. B., & Baxter, R. A. (1996). Energy efficient neural codes.
  Neural Computation, 8(3), 531-543.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import psutil
import os
import gc
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field
import warnings
import sys
import random
import io
import contextlib

# Suppress warnings
warnings.filterwarnings("ignore", message=".*_ARRAY_API not found.*")
warnings.filterwarnings("ignore", message=".*NumPy 1.x cannot be run in.*")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

# Import MNE
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core import MNE, MNEConfig


@dataclass
class FairBenchmarkResult:
    """Container for fair benchmark results with efficiency metrics."""

    model_name: str
    parameters: int
    flops_per_inference: float
    latency_per_epoch: List[float]
    memory_per_epoch: List[float]
    loss_per_epoch: List[float]
    accuracy_per_epoch: List[float]
    energy_per_epoch: List[float]  # MNE only
    neurons_per_epoch: List[int]  # MNE only
    active_neurons_per_epoch: List[int]  # MNE only
    energy_efficiency_per_epoch: List[float]  # MNE only

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
    def final_accuracy(self) -> float:
        return self.accuracy_per_epoch[-1] if self.accuracy_per_epoch else 0.0

    @property
    def accuracy_per_parameter(self) -> float:
        """Accuracy per million parameters."""
        if self.parameters > 0:
            return self.final_accuracy / (self.parameters / 1e6)
        return 0.0

    @property
    def accuracy_per_joule(self) -> float:
        """Accuracy per unit energy (MNE only)."""
        if self.energy_per_epoch and sum(self.energy_per_epoch) > 0:
            total_energy = sum(self.energy_per_epoch)
            return self.final_accuracy / total_energy
        return 0.0

    @property
    def accuracy_per_flop(self) -> float:
        """Accuracy per billion FLOPs."""
        if self.flops_per_inference > 0:
            return self.final_accuracy / (self.flops_per_inference / 1e9)
        return 0.0

    @property
    def loss_reduction_rate(self) -> float:
        """Rate of loss reduction per epoch (negative is good)."""
        if len(self.loss_per_epoch) < 2:
            return 0.0
        return (self.loss_per_epoch[-1] - self.loss_per_epoch[0]) / len(
            self.loss_per_epoch
        )


class FLOPsCounter:
    """Count FLOPs for neural network operations."""

    @staticmethod
    def count_linear_flops(in_features: int, out_features: int) -> int:
        """Count FLOPs for a linear layer: 2 * in * out (multiply + add)."""
        return 2 * in_features * out_features

    @staticmethod
    def count_activation_flops(num_elements: int) -> int:
        """Count FLOPs for activation function: num_elements."""
        return num_elements

    @staticmethod
    def count_mne_flops(
        num_neurons: int, sparsity: float = 0.8, num_steps: int = 1
    ) -> int:
        """
        Count FLOPs for MNE inference.

        MNE inference involves:
        1. Weighted sum: num_neurons^2 * (1 - sparsity) * 2 (multiply + add)
        2. Activation: num_neurons
        3. Energy computation: num_neurons * 2
        4. Resource update: num_neurons * 3

        Total per time step: num_neurons^2 * (1 - sparsity) * 2 + num_neurons * 6
        """
        synaptic_flops = int(num_neurons**2 * (1 - sparsity) * 2)
        neuron_flops = num_neurons * 6
        return (synaptic_flops + neuron_flops) * num_steps


class StandardNN(nn.Module):
    """Standard feedforward neural network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        target_flops: Optional[float] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Build layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Compute FLOPs
        self.flops_per_inference = self._compute_flops()

        # Adjust architecture if target FLOPs specified
        if target_flops is not None:
            self._adjust_for_target_flops(target_flops)

    def _compute_flops(self) -> float:
        """Compute FLOPs per inference."""
        flops = 0
        prev_dim = self.input_dim

        for hidden_dim in self.hidden_dims:
            # Linear layer
            flops += FLOPsCounter.count_linear_flops(prev_dim, hidden_dim)
            # Activation
            flops += FLOPsCounter.count_activation_flops(hidden_dim)
            prev_dim = hidden_dim

        # Output layer
        flops += FLOPsCounter.count_linear_flops(prev_dim, self.output_dim)

        return float(flops)

    def _adjust_for_target_flops(self, target_flops: float):
        """Adjust architecture to match target FLOPs."""
        current_flops = self.flops_per_inference
        if abs(current_flops - target_flops) / target_flops < 0.1:
            # Already close enough (within 10%)
            return

        # Scale hidden dimensions
        scale_factor = (target_flops / current_flops) ** 0.5
        new_hidden_dims = [max(1, int(dim * scale_factor)) for dim in self.hidden_dims]

        # Rebuild network
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in new_hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, self.output_dim))

        self.network = nn.Sequential(*layers)
        self.hidden_dims = new_hidden_dims
        self.flops_per_inference = self._compute_flops()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class EnergyConstrainedMNE(nn.Module):
    """
    MNE wrapper for energy-constrained scenarios.

    This wrapper enables MNE to:
    1. Adapt to energy constraints dynamically
    2. Track resource allocation metrics
    3. Compare fairly with other architectures
    """

    def __init__(
        self,
        mne: MNE,
        input_dim: int,
        output_dim: int,
        target_flops: Optional[float] = None,
        task_type: str = "classification",
    ):
        super().__init__()
        self.mne = mne
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type

        # Input projection
        self.input_projection = nn.Linear(input_dim, mne.config.num_neurons)

        # Output projection
        self.output_projection = nn.Linear(mne.config.num_neurons, output_dim)

        # Compute FLOPs
        self.flops_per_inference = self._compute_flops()

        # Adjust if target FLOPs specified
        if target_flops is not None:
            self._adjust_for_target_flops(target_flops)

        # Track last activations for contribution computation
        self.last_neuron_activations = None

    def _compute_flops(self) -> float:
        """Compute FLOPs per inference."""
        # Input projection
        flops = FLOPsCounter.count_linear_flops(
            self.input_dim, self.mne.config.num_neurons
        )

        # MNE core
        flops += FLOPsCounter.count_mne_flops(
            self.mne.config.num_neurons,
            sparsity=self.mne.config.sparsity,
            num_steps=1,
        )

        # Output projection
        flops += FLOPsCounter.count_linear_flops(
            self.mne.config.num_neurons, self.output_dim
        )

        return float(flops)

    def _adjust_for_target_flops(self, target_flops: float):
        """Adjust MNE to match target FLOPs."""
        current_flops = self.flops_per_inference
        if abs(current_flops - target_flops) / target_flops < 0.1:
            return

        # Scale neuron count
        scale_factor = (target_flops / current_flops) ** 0.5
        new_num_neurons = max(10, int(self.mne.config.num_neurons * scale_factor))

        # Create new config with updated neuron count
        new_config = MNEConfig(
            num_neurons=new_num_neurons,
            activation_fn=self.mne.config.activation_fn,
            kappa=self.mne.config.kappa,
            gamma=self.mne.config.gamma,
            alpha=self.mne.config.alpha,
            beta=self.mne.config.beta,
            delta=self.mne.config.delta,
            rho=self.mne.config.rho,
            target_activation=self.mne.config.target_activation,
            initial_resource=self.mne.config.initial_resource,
            initial_threshold=self.mne.config.initial_threshold,
            eta=self.mne.config.eta,
            mu=self.mne.config.mu,
            weight_init_std=self.mne.config.weight_init_std,
            weight_clip_min=self.mne.config.weight_clip_min,
            weight_clip_max=self.mne.config.weight_clip_max,
            formation_threshold=self.mne.config.formation_threshold,
            elimination_threshold=self.mne.config.elimination_threshold,
            sparsity=self.mne.config.sparsity,
            initial_energy=self.mne.config.initial_energy,
            energy_influx=self.mne.config.energy_influx,
            min_energy=self.mne.config.min_energy,
            max_energy=self.mne.config.max_energy,
            efficiency_window=self.mne.config.efficiency_window,
            history_length=self.mne.config.history_length,
            max_neurons=int(new_num_neurons * 1.5),
            min_neurons=int(new_num_neurons * 0.5),
            resource_high=self.mne.config.resource_high,
            resource_low=self.mne.config.resource_low,
            neurogenesis_rate=self.mne.config.neurogenesis_rate,
            apoptosis_rate=self.mne.config.apoptosis_rate,
            device=self.mne.config.device,
        )

        # Rebuild MNE with new configuration
        self.mne = MNE(new_config)

        # Rebuild projections
        self.input_projection = nn.Linear(self.input_dim, new_num_neurons)
        self.output_projection = nn.Linear(new_num_neurons, self.output_dim)

        # Recompute FLOPs
        self.flops_per_inference = self._compute_flops()

    def forward(
        self,
        x: torch.Tensor,
        state=None,
        apply_plasticity: bool = False,
        contribution=None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Forward pass."""
        batch_size = x.shape[0]

        # Initialize state if not provided
        if state is None:
            state = self.mne.get_initial_state(batch_size=batch_size)

        # Project input
        projected_input = self.input_projection(x)

        # MNE forward pass
        neuron_activations, new_state = self.mne.forward(
            projected_input,
            state,
            contribution=contribution,
            apply_plasticity=apply_plasticity,
        )

        # Store activations for contribution computation
        self.last_neuron_activations = neuron_activations.detach()

        # Project to output
        outputs = self.output_projection(neuron_activations)

        return outputs, new_state

    def compute_contribution(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient-based contribution."""
        if self.last_neuron_activations is None:
            batch_size = outputs.shape[0]
            return torch.ones(
                batch_size, self.mne.config.num_neurons, device=outputs.device
            )

        neuron_activations = self.last_neuron_activations.clone()
        neuron_activations.requires_grad_(True)

        outputs_with_grad = self.output_projection(neuron_activations)

        # Use appropriate loss function based on task type
        if self.task_type == "classification":
            loss_recomputed = nn.CrossEntropyLoss()(outputs_with_grad, targets)
        else:
            loss_recomputed = nn.MSELoss()(outputs_with_grad, targets)

        grad = torch.autograd.grad(
            loss_recomputed,
            neuron_activations,
            create_graph=False,
            retain_graph=False,
        )[0]

        return torch.abs(grad)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0.0


def create_energy_constrained_dataset(
    num_samples: int = 1000,
    input_dim: int = 20,
    output_dim: int = 5,
    seed: int = 42,
    task_type: str = "classification",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create dataset for energy-constrained scenarios.

    These tasks are designed to:
    1. Not favor any specific architecture
    2. Benefit from adaptive resource allocation
    3. Be solvable with limited computational budget
    """
    random.seed(seed)
    torch.manual_seed(seed)

    X_list = []
    y_list = []

    for _ in range(num_samples):
        # Generate input features
        row = [random.gauss(0, 1) for _ in range(input_dim)]
        X_list.append(row)

    with contextlib.redirect_stderr(io.StringIO()):
        X = torch.tensor(X_list, dtype=torch.float32)

        if task_type == "classification":
            # Create classification task with clear but learnable patterns
            # Use a simple linear decision boundary with noise
            W = torch.randn(input_dim, output_dim) * 0.5
            logits = torch.mm(X, W)
            y = torch.argmax(logits, dim=1)
        else:
            # Regression task with smooth function
            W = torch.randn(input_dim, output_dim) * 0.3
            y = torch.mm(X, W)
            y = torch.tanh(y)  # Bounded output

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
    task_type: str = "classification",
) -> FairBenchmarkResult:
    """Train StandardNN and collect metrics."""
    print(f"\nTraining StandardNN...")

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if task_type == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    result = FairBenchmarkResult(
        model_name="StandardNN",
        parameters=count_parameters(model),
        flops_per_inference=model.flops_per_inference,
        latency_per_epoch=[],
        memory_per_epoch=[],
        loss_per_epoch=[],
        accuracy_per_epoch=[],
        energy_per_epoch=[],
        neurons_per_epoch=[],
        active_neurons_per_epoch=[],
        energy_efficiency_per_epoch=[],
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

            # Compute accuracy
            if task_type == "classification":
                val_pred = torch.argmax(val_output, dim=1)
                val_acc = (val_pred == y_val).float().mean().item()
            else:
                # For regression, use negative MSE as "accuracy" metric
                val_acc = -val_loss

        result.latency_per_epoch.append(epoch_time)
        result.loss_per_epoch.append(val_loss)
        result.accuracy_per_epoch.append(val_acc)

        print(
            f"  Epoch {epoch + 1}/{num_epochs}: "
            f"Loss={val_loss:.4f}, Acc={val_acc:.4f}, "
            f"Time={epoch_time:.2f}s, Memory={result.memory_per_epoch[-1]:.1f}MB"
        )

    return result


def train_mne(
    model: EnergyConstrainedMNE,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    task_type: str = "classification",
    energy_constraint: Optional[float] = None,
) -> FairBenchmarkResult:
    """
    Train MNE with energy constraints.

    Args:
        energy_constraint: If set, limit energy influx to simulate constrained environment
    """
    print(f"\nTraining MNE (energy-constrained)...")

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Count all parameters
    total_params = count_parameters(model)

    result = FairBenchmarkResult(
        model_name="MNE",
        parameters=total_params,
        flops_per_inference=model.flops_per_inference,
        latency_per_epoch=[],
        memory_per_epoch=[],
        loss_per_epoch=[],
        accuracy_per_epoch=[],
        energy_per_epoch=[],
        neurons_per_epoch=[],
        active_neurons_per_epoch=[],
        energy_efficiency_per_epoch=[],
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if task_type == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    # Apply energy constraint if specified
    if energy_constraint is not None:
        initial_state = model.mne.get_initial_state(batch_size=1)
        initial_state = model.mne.set_energy_influx(initial_state, energy_constraint)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start_memory = get_memory_usage()

            # First forward pass (no plasticity)
            outputs, state = model.forward(data, apply_plasticity=False)

            # Compute loss
            loss = criterion(outputs, target)

            # Compute contribution
            contribution = model.compute_contribution(outputs, target)

            # Second forward pass (with plasticity)
            outputs, new_state = model.forward(
                data, state=state, contribution=contribution, apply_plasticity=True
            )

            # Backward pass
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

        # Get MNE metrics
        metrics = model.mne.get_metrics(new_state)

        # Validation
        model.mne.eval()
        with torch.no_grad():
            val_outputs, _ = model.forward(X_val, apply_plasticity=False)
            val_loss = criterion(val_outputs, y_val).item()

            if task_type == "classification":
                val_pred = torch.argmax(val_outputs, dim=1)
                val_acc = (val_pred == y_val).float().mean().item()
            else:
                val_acc = -val_loss

        result.latency_per_epoch.append(epoch_time)
        result.loss_per_epoch.append(val_loss)
        result.accuracy_per_epoch.append(val_acc)
        result.energy_per_epoch.append(metrics.get("total_energy", 0.0))
        result.neurons_per_epoch.append(metrics.get("num_neurons", 0))
        result.active_neurons_per_epoch.append(metrics.get("num_active", 0))
        result.energy_efficiency_per_epoch.append(metrics.get("efficiency", 0.0))

        print(
            f"  Epoch {epoch + 1}/{num_epochs}: "
            f"Loss={val_loss:.4f}, Acc={val_acc:.4f}, "
            f"Time={epoch_time:.2f}s, Memory={result.memory_per_epoch[-1]:.1f}MB, "
            f"Neurons={metrics.get('num_neurons', 0)}, "
            f"Energy={metrics.get('total_energy', 0.0):.1f}, "
            f"Eff={metrics.get('efficiency', 0.0):.4f}"
        )

    return result


def run_fair_benchmark(
    num_epochs: int = 20,
    dataset_size: int = 1000,
    input_dim: int = 20,
    hidden_dim: int = 64,
    output_dim: int = 5,
    target_flops: float = 10000.0,  # Target FLOPs per inference
    seed: int = 42,
    task_type: str = "classification",
    energy_constraint: Optional[float] = None,
) -> Dict[str, FairBenchmarkResult]:
    """
    Run fair benchmark with matched computational budgets.

    Key features:
    1. All architectures matched to target_flops
    2. Energy-constrained scenarios for MNE
    3. Efficiency metrics computed
    """
    print("=" * 80)
    print("FAIR COMPARISON BENCHMARK")
    print("Architectures matched by FLOPs, not parameters")
    print("=" * 80)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create dataset
    print(f"\nCreating energy-constrained dataset...")
    X, y = create_energy_constrained_dataset(
        dataset_size, input_dim, output_dim, seed, task_type
    )

    split_idx = int(0.8 * dataset_size)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(
        f"Dataset: {dataset_size} samples, {input_dim} features, {output_dim} outputs"
    )
    print(f"Target FLOPs per inference: {target_flops:,.0f}")
    print(f"Train: {split_idx} samples, Validation: {dataset_size - split_idx} samples")

    # Initialize models with matched FLOPs
    print(f"\nInitializing models with matched FLOPs...")

    # StandardNN
    standard_nn = StandardNN(
        input_dim, [hidden_dim, hidden_dim], output_dim, target_flops=target_flops
    )
    print(
        f"StandardNN: {count_parameters(standard_nn):,} params, "
        f"{standard_nn.flops_per_inference:,.0f} FLOPs"
    )

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
        energy_influx=energy_constraint if energy_constraint else 10.0,
        min_energy=20.0,
        max_energy=200.0,
        max_neurons=int(hidden_dim * 1.5),
        min_neurons=int(hidden_dim * 0.5),
        resource_high=2.0,
        resource_low=0.1,
        device="cpu",
    )
    mne = MNE(mne_config)
    energy_constrained_mne = EnergyConstrainedMNE(
        mne, input_dim, output_dim, target_flops=target_flops, task_type=task_type
    )
    print(
        f"MNE: {count_parameters(energy_constrained_mne):,} params, "
        f"{energy_constrained_mne.flops_per_inference:,.0f} FLOPs"
    )

    gc.collect()

    # Train models
    print(f"\nStarting benchmark with {num_epochs} epochs...")

    results = {}

    results["standard_nn"] = train_standard_nn(
        standard_nn,
        X_train,
        y_train,
        X_val,
        y_val,
        num_epochs=num_epochs,
        batch_size=32,
        learning_rate=0.01,
        task_type=task_type,
    )
    gc.collect()

    results["mne"] = train_mne(
        energy_constrained_mne,
        X_train,
        y_train,
        X_val,
        y_val,
        num_epochs=num_epochs,
        batch_size=32,
        learning_rate=0.01,
        task_type=task_type,
        energy_constraint=energy_constraint,
    )

    return results


def analyze_fair_results(results: Dict[str, FairBenchmarkResult]):
    """Analyze fair comparison results with efficiency metrics."""
    print("\n" + "=" * 80)
    print("FAIR COMPARISON ANALYSIS")
    print("=" * 80)

    # Print computational budget matching
    print(f"\nComputational Budget Matching:")
    print(f"{'Model':<15} {'Params':<12} {'FLOPs':<15} {'FLOPs Match':<12}")
    print("-" * 60)

    target_flops = results["standard_nn"].flops_per_inference
    for name, result in results.items():
        flops_match = abs(result.flops_per_inference - target_flops) / target_flops
        print(
            f"{name:<15} {result.parameters:<12,} "
            f"{result.flops_per_inference:<15,.0f} {flops_match:<12.2%}"
        )

    # Performance comparison
    print(f"\nPerformance Comparison:")
    print(
        f"{'Model':<15} {'Accuracy':<12} {'Loss':<12} {'Latency(s)':<12} {'Memory(MB)':<12}"
    )
    print("-" * 75)

    for name, result in results.items():
        print(
            f"{name:<15} {result.final_accuracy:<12.4f} "
            f"{result.final_loss:<12.4f} {result.avg_latency:<12.3f} "
            f"{result.avg_memory:<12.1f}"
        )

    # Efficiency metrics
    print(f"\nEfficiency Metrics:")
    print(f"{'Model':<15} {'Acc/Param':<15} {'Acc/FLOP':<15} {'Acc/Joule':<15}")
    print("-" * 65)

    for name, result in results.items():
        print(
            f"{name:<15} {result.accuracy_per_parameter:<15.4f} "
            f"{result.accuracy_per_flop:<15.6f} {result.accuracy_per_joule:<15.6f}"
        )

    # MNE-specific metrics
    if "mne" in results:
        mne_result = results["mne"]
        print(f"\nMNE-Specific Metrics:")
        print(
            f"  Final Neurons: {mne_result.neurons_per_epoch[-1] if mne_result.neurons_per_epoch else 'N/A'}"
        )
        print(
            f"  Active Neurons: {mne_result.active_neurons_per_epoch[-1] if mne_result.active_neurons_per_epoch else 'N/A'}"
        )
        print(f"  Total Energy: {sum(mne_result.energy_per_epoch):.1f}")
        print(
            f"  Avg Efficiency: {np.mean(mne_result.energy_efficiency_per_epoch):.4f}"
        )

        # Dynamic resource allocation
        if mne_result.neurons_per_epoch:
            neuron_change = (
                mne_result.neurons_per_epoch[-1] - mne_result.neurons_per_epoch[0]
            )
            print(f"  Neuron Change: {neuron_change:+d}")
            if neuron_change > 0:
                print(f"    → Neurogenesis: MNE adapted by growing neurons")
            elif neuron_change < 0:
                print(f"    → Apoptosis: MNE adapted by pruning neurons")

    # Overall efficiency ranking
    print(f"\nEfficiency Ranking (higher is better):")

    # Parameter efficiency
    param_efficiency = {
        name: result.accuracy_per_parameter for name, result in results.items()
    }
    sorted_param = sorted(param_efficiency.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Parameter Efficiency (Accuracy per million params):")
    for rank, (name, score) in enumerate(sorted_param, 1):
        print(f"    {rank}. {name}: {score:.4f}")

    # FLOP efficiency
    flop_efficiency = {
        name: result.accuracy_per_flop for name, result in results.items()
    }
    sorted_flop = sorted(flop_efficiency.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  FLOP Efficiency (Accuracy per billion FLOPs):")
    for rank, (name, score) in enumerate(sorted_flop, 1):
        print(f"    {rank}. {name}: {score:.6f}")

    # Energy efficiency (MNE only)
    if "mne" in results:
        print(f"\n  Energy Efficiency (Accuracy per joule):")
        print(f"    MNE: {results['mne'].accuracy_per_joule:.6f}")
        print(f"    (StandardNN does not track energy)")

    # Key insights
    print(f"\n{'=' * 40}")
    print("KEY INSIGHTS")
    print(f"{'=' * 40}")

    # Compare accuracy at matched FLOPs
    std_acc = results["standard_nn"].final_accuracy
    mne_acc = results["mne"].final_accuracy

    if mne_acc > std_acc * 1.05:
        print(
            f"• MNE achieves {100 * (mne_acc / std_acc - 1):.1f}% higher accuracy "
            f"at matched FLOPs"
        )
    elif std_acc > mne_acc * 1.05:
        print(
            f"• StandardNN achieves {100 * (std_acc / mne_acc - 1):.1f}% higher accuracy "
            f"at matched FLOPs"
        )
    else:
        print(f"• Both models achieve similar accuracy at matched FLOPs")

    # Compare parameter efficiency
    std_param_eff = results["standard_nn"].accuracy_per_parameter
    mne_param_eff = results["mne"].accuracy_per_parameter

    if mne_param_eff > std_param_eff:
        print(
            f"• MNE is {100 * (mne_param_eff / std_param_eff - 1):.1f}% more "
            f"parameter-efficient"
        )
    else:
        print(
            f"• StandardNN is {100 * (std_param_eff / mne_param_eff - 1):.1f}% more "
            f"parameter-efficient"
        )

    # MNE's adaptive capabilities
    if "mne" in results and results["mne"].neurons_per_epoch:
        neuron_change = (
            results["mne"].neurons_per_epoch[-1] - results["mne"].neurons_per_epoch[0]
        )
        if abs(neuron_change) > 0:
            print(
                f"• MNE demonstrated adaptive resource allocation "
                f"({neuron_change:+d} neurons)"
            )


def plot_fair_results(
    results: Dict[str, FairBenchmarkResult],
    save_path: str = "fair_comparison_results.png",
):
    """Plot fair comparison results."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        colors = {"standard_nn": "blue", "mne": "red"}

        # Plot 1: Accuracy over epochs
        ax = axes[0, 0]
        for name, result in results.items():
            epochs = range(1, len(result.accuracy_per_epoch) + 1)
            ax.plot(
                epochs,
                result.accuracy_per_epoch,
                label=name,
                color=colors.get(name, "black"),
                linewidth=2,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Loss over epochs
        ax = axes[0, 1]
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
        ax.set_title("Loss Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Latency over epochs
        ax = axes[0, 2]
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
        ax.set_title("Training Latency")
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
                label="Total Neurons",
            )
            if results["mne"].active_neurons_per_epoch:
                ax.plot(
                    epochs,
                    results["mne"].active_neurons_per_epoch,
                    color=colors["mne"],
                    linewidth=2,
                    linestyle="--",
                    label="Active Neurons",
                )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Neurons")
            ax.set_title("MNE Dynamic Resource Allocation")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 6: Efficiency comparison bar chart
        ax = axes[1, 2]
        metrics = ["Accuracy", "Acc/Param", "Acc/FLOP"]
        x = np.arange(len(metrics))
        width = 0.35

        std_values = [
            results["standard_nn"].final_accuracy,
            results["standard_nn"].accuracy_per_parameter,
            results["standard_nn"].accuracy_per_flop,
        ]
        mne_values = [
            results["mne"].final_accuracy,
            results["mne"].accuracy_per_parameter,
            results["mne"].accuracy_per_flop,
        ]

        ax.bar(
            x - width / 2,
            std_values,
            width,
            label="StandardNN",
            color=colors["standard_nn"],
            alpha=0.7,
        )
        ax.bar(
            x + width / 2,
            mne_values,
            width,
            label="MNE",
            color=colors["mne"],
            alpha=0.7,
        )
        ax.set_xlabel("Metric")
        ax.set_ylabel("Value")
        ax.set_title("Efficiency Comparison")
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
    print("Fair Comparison Benchmark")
    print("Comparing architectures with matched computational budgets")

    try:
        # Run fair benchmark
        results = run_fair_benchmark(
            num_epochs=20,
            dataset_size=1000,
            input_dim=20,
            hidden_dim=64,
            output_dim=5,
            target_flops=10000.0,  # Match FLOPs across models
            seed=42,
            task_type="classification",
            energy_constraint=5.0,  # Constrain energy to test MNE's adaptation
        )

        # Analyze results
        analyze_fair_results(results)

        # Plot results
        print("\nGenerating plots...")
        plot_fair_results(results)
        print("Plots generated successfully!")

        print("\n" + "=" * 80)
        print("FAIR BENCHMARK COMPLETE")
        print("=" * 80)

    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
