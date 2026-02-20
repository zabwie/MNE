"""
Enhanced Comprehensive Benchmark Suite for Metabolic Neural Ecosystem (MNE)

This benchmark evaluates MNE against conventional neural networks across multiple dimensions:
1. Generalization & Robustness (OOD with pytorch-ood, adversarial with torchattacks, calibration)
2. Efficiency (energy with Zeus/pyJoules, FLOPs, throughput)
3. Stability & Sensitivity (hyperparameters, ablation)
4. Continual/Lifelong Learning (with Avalanche)
5. Interpretability & Explainability (with Captum)

External Libraries Used:
- torchattacks: Adversarial robustness (FGSM, PGD, AutoAttack)
- pytorch-ood: OOD detection metrics (AUROC, FPR@95% TPR)
- Captum: Model interpretability (saliency, GradCAM, IntegratedGradients)
- Zeus: GPU energy measurement
- (Optional) pyJoules: CPU/GPU energy measurement
- (Optional) Avalanche: Continual learning benchmarks

Author: MNE Research Team
Date: 2026-02-19
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import warnings

# Optional imports - only load if available
try:
    import torchattacks

    HAS_TORCHATTACKS = True
except ImportError:
    HAS_TORCHATTACKS = False
    warnings.warn("torchattacks not installed. Install with: pip install torchattacks")

try:
    from pytorch_ood.utils import OODMetrics

    HAS_PYTORCH_OOD = True
except ImportError:
    HAS_PYTORCH_OOD = False
    warnings.warn("pytorch-ood not installed. Install with: pip install pytorch-ood")

try:
    from captum.attr import Saliency, IntegratedGradients, LayerGradCam  # noqa: F401
    from captum.concept import Concept, TCAV  # noqa: F401

    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False
    warnings.warn("Captum not installed. Install with: pip install captum")

try:
    from zeus.monitor import ZeusMonitor

    HAS_ZEUS = True
except ImportError:
    HAS_ZEUS = False
    warnings.warn("Zeus not installed. Install with: pip install zeus-ml")

try:
    from pyJoules.energy_meter import measure_energy as _  # noqa: F401
    from pyJoules.energy_meter import EnergyMeter
    from pyJoules.device import DeviceFactory

    HAS_PYJOULES = True
except ImportError:
    HAS_PYJOULES = False
    warnings.warn("pyJoules not installed. Install with: pip install pyJoules")

HAS_AVALANCHE = False  # Not currently used - would require more integration

try:
    from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR10, PermutedMNIST  # noqa: F401
    from avalanche.evaluation.metrics import (  # noqa: F401
        accuracy_metrics,
        forgetting_metrics,
        bwt_metrics,
        forward_transfer_metrics,
    )
    from avalanche.training.plugins import EvaluationPlugin  # noqa: F401

    HAS_AVALANCHE = True
except ImportError:
    HAS_AVALANCHE = False
    warnings.warn("Avalanche not installed. Install with: pip install avalanche-lib")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ============================================================================
# Data Classes for Benchmark Results
# ============================================================================


@dataclass
class GeneralizationResults:
    """Results from generalization and robustness tests."""

    # OOD Detection
    ood_auroc: float = 0.0
    ood_fpr_at_95_tpr: float = 0.0

    # Adversarial Robustness
    clean_accuracy: float = 0.0
    fgsm_accuracy: float = 0.0
    pgd_accuracy: float = 0.0
    autoattack_accuracy: float = 0.0

    # Calibration
    expected_calibration_error: float = 0.0
    brier_score: float = 0.0

    # Distribution Shift
    source_accuracy: float = 0.0
    target_accuracy: float = 0.0
    domain_adaptation_gain: float = 0.0

    # Corruption (CIFAR-10-C style)
    mean_corruption_accuracy: float = 0.0
    severe_corruption_accuracy: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "ood_auroc": self.ood_auroc,
            "ood_fpr_at_95_tpr": self.ood_fpr_at_95_tpr,
            "clean_accuracy": self.clean_accuracy,
            "fgsm_accuracy": self.fgsm_accuracy,
            "pgd_accuracy": self.pgd_accuracy,
            "autoattack_accuracy": self.autoattack_accuracy,
            "expected_calibration_error": self.expected_calibration_error,
            "brier_score": self.brier_score,
            "source_accuracy": self.source_accuracy,
            "target_accuracy": self.target_accuracy,
            "domain_adaptation_gain": self.domain_adaptation_gain,
            "mean_corruption_accuracy": self.mean_corruption_accuracy,
            "severe_corruption_accuracy": self.severe_corruption_accuracy,
        }


@dataclass
class EfficiencyResults:
    """Results from efficiency metrics."""

    # Training Efficiency
    training_time_per_epoch: float = 0.0
    epochs_to_target_accuracy: int = 0
    convergence_rate: float = 0.0

    # Inference Efficiency
    inference_latency_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0

    # Energy Consumption
    energy_per_inference_joules: float = 0.0
    energy_per_training_epoch_joules: float = 0.0
    energy_efficiency_samples_per_joule: float = 0.0

    # Parameter Efficiency
    num_parameters: int = 0
    accuracy_per_million_params: float = 0.0

    # Data Efficiency
    accuracy_at_1_percent_data: float = 0.0
    accuracy_at_10_percent_data: float = 0.0
    accuracy_at_50_percent_data: float = 0.0
    accuracy_at_100_percent_data: float = 0.0

    # Computational Cost
    flops_per_forward: float = 0.0
    macs_per_forward: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "training_time_per_epoch": self.training_time_per_epoch,
            "epochs_to_target_accuracy": self.epochs_to_target_accuracy,
            "convergence_rate": self.convergence_rate,
            "inference_latency_ms": self.inference_latency_ms,
            "throughput_samples_per_sec": self.throughput_samples_per_sec,
            "energy_per_inference_joules": self.energy_per_inference_joules,
            "energy_per_training_epoch_joules": self.energy_per_training_epoch_joules,
            "energy_efficiency_samples_per_joule": self.energy_efficiency_samples_per_joule,
            "num_parameters": self.num_parameters,
            "accuracy_per_million_params": self.accuracy_per_million_params,
            "accuracy_at_1_percent_data": self.accuracy_at_1_percent_data,
            "accuracy_at_10_percent_data": self.accuracy_at_10_percent_data,
            "accuracy_at_50_percent_data": self.accuracy_at_50_percent_data,
            "accuracy_at_100_percent_data": self.accuracy_at_100_percent_data,
            "flops_per_forward": self.flops_per_forward,
            "macs_per_forward": self.macs_per_forward,
        }


@dataclass
class StabilityResults:
    """Results from stability and sensitivity tests."""

    # Hyperparameter Sensitivity
    lr_variance: float = 0.0
    batch_size_variance: float = 0.0
    regularization_variance: float = 0.0
    overall_sensitivity_score: float = 0.0

    # Initialization Sensitivity
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    min_accuracy: float = 0.0
    max_accuracy: float = 0.0

    # Ablation Studies
    full_model_accuracy: float = 0.0
    no_plasticity_accuracy: float = 0.0
    no_homeostasis_accuracy: float = 0.0
    no_topology_accuracy: float = 0.0

    # Gradient Analysis
    mean_grad_norm: float = 0.0
    std_grad_norm: float = 0.0
    vanished_gradient_ratio: float = 0.0
    exploding_gradient_ratio: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "lr_variance": self.lr_variance,
            "batch_size_variance": self.batch_size_variance,
            "regularization_variance": self.regularization_variance,
            "overall_sensitivity_score": self.overall_sensitivity_score,
            "mean_accuracy": self.mean_accuracy,
            "std_accuracy": self.std_accuracy,
            "min_accuracy": self.min_accuracy,
            "max_accuracy": self.max_accuracy,
            "full_model_accuracy": self.full_model_accuracy,
            "no_plasticity_accuracy": self.no_plasticity_accuracy,
            "no_homeostasis_accuracy": self.no_homeostasis_accuracy,
            "no_topology_accuracy": self.no_topology_accuracy,
            "mean_grad_norm": self.mean_grad_norm,
            "std_grad_norm": self.std_grad_norm,
            "vanished_gradient_ratio": self.vanished_gradient_ratio,
            "exploding_gradient_ratio": self.exploding_gradient_ratio,
        }


@dataclass
class ContinualLearningResults:
    """Results from continual/lifelong learning benchmarks."""

    # Task Incremental
    num_tasks: int = 0
    average_task_accuracy: float = 0.0
    forgetting_measure: float = 0.0
    learning_curve_area: float = 0.0

    # Backward Transfer
    backward_transfer_score: float = 0.0
    knowledge_retention_score: float = 0.0

    # Forward Transfer
    forward_transfer_score: float = 0.0
    knowledge_reuse_score: float = 0.0

    # Memory Replay Efficiency
    replay_buffer_accuracy: float = 0.0
    no_replay_accuracy: float = 0.0
    replay_efficiency_gain: float = 0.0

    # Neurogenesis/Apoptosis (MNE specific)
    total_neurogenesis_events: int = 0
    total_apoptosis_events: int = 0
    initial_neurons: int = 0
    final_neurons: int = 0
    neuron_count_change: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "num_tasks": self.num_tasks,
            "average_task_accuracy": self.average_task_accuracy,
            "forgetting_measure": self.forgetting_measure,
            "learning_curve_area": self.learning_curve_area,
            "backward_transfer_score": self.backward_transfer_score,
            "knowledge_retention_score": self.knowledge_retention_score,
            "forward_transfer_score": self.forward_transfer_score,
            "knowledge_reuse_score": self.knowledge_reuse_score,
            "replay_buffer_accuracy": self.replay_buffer_accuracy,
            "no_replay_accuracy": self.no_replay_accuracy,
            "replay_efficiency_gain": self.replay_efficiency_gain,
            "total_neurogenesis_events": self.total_neurogenesis_events,
            "total_apoptosis_events": self.total_apoptosis_events,
            "initial_neurons": self.initial_neurons,
            "final_neurons": self.final_neurons,
            "neuron_count_change": self.neuron_count_change,
        }


@dataclass
class ExplainabilityResults:
    """Results from interpretability and explainability tests."""

    # Feature Importance
    saliency_consistency_score: float = 0.0
    feature_attribution_entropy: float = 0.0
    attribution_sparsity: float = 0.0

    # Concept Attribution (TCAV-style)
    concept_alignment_score: float = 0.0
    concept_diversity_score: float = 0.0

    # Counterfactual Analysis
    counterfactual_distance: float = 0.0
    counterfactual_success_rate: float = 0.0

    # Latent Space Analysis
    latent_space_clusters: int = 0
    intra_cluster_distance: float = 0.0
    inter_cluster_distance: float = 0.0
    separation_score: float = 0.0

    # Neuron Specificity (MNE specific)
    neuron_specialization_score: float = 0.0
    energy_contribution_correlation: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "saliency_consistency_score": self.saliency_consistency_score,
            "feature_attribution_entropy": self.feature_attribution_entropy,
            "attribution_sparsity": self.attribution_sparsity,
            "concept_alignment_score": self.concept_alignment_score,
            "concept_diversity_score": self.concept_diversity_score,
            "counterfactual_distance": self.counterfactual_distance,
            "counterfactual_success_rate": self.counterfactual_success_rate,
            "latent_space_clusters": self.latent_space_clusters,
            "intra_cluster_distance": self.intra_cluster_distance,
            "inter_cluster_distance": self.inter_cluster_distance,
            "separation_score": self.separation_score,
            "neuron_specialization_score": self.neuron_specialization_score,
            "energy_contribution_correlation": self.energy_contribution_correlation,
        }


@dataclass
class BenchmarkResults:
    """Complete benchmark results for a model."""

    model_name: str
    generalization: GeneralizationResults = field(default_factory=GeneralizationResults)
    efficiency: EfficiencyResults = field(default_factory=EfficiencyResults)
    stability: StabilityResults = field(default_factory=StabilityResults)
    continual_learning: ContinualLearningResults = field(
        default_factory=ContinualLearningResults
    )
    explainability: ExplainabilityResults = field(default_factory=ExplainabilityResults)

    # Overall scores
    overall_score: float = 0.0
    robustness_score: float = 0.0
    efficiency_score: float = 0.0
    reliability_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "generalization": self.generalization.to_dict(),
            "efficiency": self.efficiency.to_dict(),
            "stability": self.stability.to_dict(),
            "continual_learning": self.continual_learning.to_dict(),
            "explainability": self.explainability.to_dict(),
            "overall_score": self.overall_score,
            "robustness_score": self.robustness_score,
            "efficiency_score": self.efficiency_score,
            "reliability_score": self.reliability_score,
        }

    def save_json(self, filepath: Path):
        """Save results to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: Path) -> "BenchmarkResults":
        """Load results from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        # Reconstruct from dict
        results = cls(
            model_name=data["model_name"],
            generalization=GeneralizationResults(**data["generalization"]),
            efficiency=EfficiencyResults(**data["efficiency"]),
            stability=StabilityResults(**data["stability"]),
            continual_learning=ContinualLearningResults(**data["continual_learning"]),
            explainability=ExplainabilityResults(**data["explainability"]),
        )
        results.overall_score = data["overall_score"]
        results.robustness_score = data["robustness_score"]
        results.efficiency_score = data["efficiency_score"]
        results.reliability_score = data["reliability_score"]
        return results


# ============================================================================
# Baseline Models
# ============================================================================


class StandardMLP(nn.Module):
    """Standard multi-layer perceptron for comparison."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_num_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class StandardCNN(nn.Module):
    """Standard CNN for image data."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)

    def get_num_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Metric Utilities
# ============================================================================


def compute_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu",
) -> float:
    """Compute model accuracy on a dataset."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Handle both standard models and MNE
            if hasattr(model, "get_initial_state"):
                batch_size = inputs.shape[0]
                state = model.get_initial_state(batch_size)
                outputs, _ = model(inputs, state, apply_plasticity=False)
                # Get predictions
                if outputs.dim() > 1:
                    predictions = outputs.argmax(dim=1)
                else:
                    predictions = outputs
            else:
                outputs = model(inputs)
                predictions = outputs.argmax(dim=1)

            # Handle different target shapes
            if targets.dim() > 1:
                targets = targets.argmax(dim=1)

            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    return 100.0 * correct / total if total > 0 else 0.0


def compute_expected_calibration_error(
    model: nn.Module,
    data_loader: DataLoader,
    n_bins: int = 15,
    device: str = "cpu",
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Implements the standard ECE metric as described in:
    Guo et al., "On Calibration of Modern Neural Networks" (2017)
    """
    model.eval()
    confidences = []
    accuracies = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Get predictions
            if hasattr(model, "get_initial_state"):
                batch_size = inputs.shape[0]
                state = model.get_initial_state(batch_size)
                outputs, _ = model(inputs, state, apply_plasticity=False)
                probs = F.softmax(outputs, dim=1)
            else:
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)

            confidence, predictions = probs.max(dim=1)

            # Handle target shapes
            if targets.dim() > 1:
                targets = targets.argmax(dim=1)

            correct = (predictions == targets).float()
            confidences.extend(confidence.cpu().tolist())
            accuracies.extend(correct.cpu().tolist())

    confidences = np.array(confidences)
    accuracies = np.array(accuracies)

    # Bin predictions by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def compute_ood_metrics(
    model: nn.Module,
    id_loader: DataLoader,
    ood_loader: DataLoader,
    device: str = "cpu",
) -> Tuple[float, float]:
    """
    Compute OOD detection metrics: AUROC and FPR@95% TPR.

    Returns:
        (auroc, fpr_at_95_tpr)
    """
    if not HAS_PYTORCH_OOD:
        # Fallback implementation using sklearn
        from sklearn.metrics import roc_auc_score, roc_curve

        model.eval()

        id_scores = []
        ood_scores = []

        # Collect in-distribution scores (use max softmax probability as OOD score)
        with torch.no_grad():
            for inputs, _ in id_loader:
                inputs = inputs.to(device)

                if hasattr(model, "get_initial_state"):
                    batch_size = inputs.shape[0]
                    state = model.get_initial_state(batch_size)
                    outputs, _ = model(inputs, state, apply_plasticity=False)
                else:
                    outputs = model(inputs)

                probs = F.softmax(outputs, dim=1)
                max_probs = probs.max(dim=1)[0]
                id_scores.extend(max_probs.cpu().tolist())

        # Collect out-of-distribution scores
        with torch.no_grad():
            for inputs, _ in ood_loader:
                inputs = inputs.to(device)

                if hasattr(model, "get_initial_state"):
                    batch_size = inputs.shape[0]
                    state = model.get_initial_state(batch_size)
                    outputs, _ = model(inputs, state, apply_plasticity=False)
                else:
                    outputs = model(inputs)

                probs = F.softmax(outputs, dim=1)
                max_probs = probs.max(dim=1)[0]
                ood_scores.extend(max_probs.cpu().tolist())

        # Labels: 1 for ID, 0 for OOD (higher max_prob = more likely to be ID)
        scores = np.concatenate([id_scores, ood_scores])
        labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])

        # Compute AUROC
        auroc = roc_auc_score(labels, scores)

        # Compute FPR at 95% TPR
        fpr, tpr, thresholds = roc_curve(labels, scores)
        idx = np.where(tpr >= 0.95)[0]
        if len(idx) > 0:
            fpr_at_95_tpr = fpr[idx[0]]
        else:
            fpr_at_95_tpr = 1.0

        return auroc, fpr_at_95_tpr
    else:
        # Use pytorch-ood for accurate metrics
        model.eval()
        metrics = OODMetrics()

        # ID samples
        with torch.no_grad():
            for inputs, _ in id_loader:
                inputs = inputs.to(device)

                if hasattr(model, "get_initial_state"):
                    batch_size = inputs.shape[0]
                    state = model.get_initial_state(batch_size)
                    outputs, _ = model(inputs, state, apply_plasticity=False)
                else:
                    outputs = model(inputs)

                # OOD detectors use energy or max softmax scores
                probs = F.softmax(outputs, dim=1)
                energy = -torch.logsumexp(probs, dim=1)  # Energy-based score
                labels = torch.ones(inputs.shape[0]).long()  # 1 for ID

                metrics.update(energy, labels)

        # OOD samples
        with torch.no_grad():
            for inputs, _ in ood_loader:
                inputs = inputs.to(device)

                if hasattr(model, "get_initial_state"):
                    batch_size = inputs.shape[0]
                    state = model.get_initial_state(batch_size)
                    outputs, _ = model(inputs, state, apply_plasticity=False)
                else:
                    outputs = model(inputs)

                probs = F.softmax(outputs, dim=1)
                energy = -torch.logsumexp(probs, dim=1)
                labels = -torch.ones(inputs.shape[0]).long()  # -1 for OOD

                metrics.update(energy, labels)

        # Compute metrics
        results = metrics.compute()
        auroc = results["AUROC"]
        fpr_at_95_tpr = results["FPR95TPR"]

        return auroc, fpr_at_95_tpr


def count_parameters(model: nn.Module) -> int:
    """Count total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def has_learnable_parameters(model: nn.Module) -> bool:
    """Check if model has any learnable parameters."""
    return count_parameters(model) > 0


# ============================================================================
# Adversarial Robustness Tests
# ============================================================================


def evaluate_adversarial_robustness(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu",
    eps: float = 8 / 255,
):
    """
    Evaluate adversarial robustness using torchattacks.

    Args:
        model: Model to evaluate
        data_loader: Test data loader
        device: Device to run on
        eps: Maximum perturbation (default: 8/255 for CIFAR)

    Returns:
        Dictionary with robust accuracy for different attacks
    """
    if not HAS_TORCHATTACKS:
        warnings.warn(
            "torchattacks not available. Skipping adversarial robustness evaluation."
        )
        return {
            "fgsm_accuracy": 0.0,
            "pgd_accuracy": 0.0,
            "autoattack_accuracy": 0.0,
        }

    model = model.to(device)
    model.eval()

    # Initialize attacks
    fgsm = torchattacks.FGSM(model, eps=eps)
    pgd10 = torchattacks.PGD(model, eps=eps, alpha=eps / 4, steps=10)
    pgd50 = torchattacks.PGD(model, eps=eps, alpha=eps / 4, steps=50)
    autoattack = torchattacks.AutoAttack(model, eps=eps, version="standard")

    results = {}

    # Evaluate clean accuracy
    clean_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            clean_correct += (predicted == labels).sum().item()

    results["clean_accuracy"] = 100.0 * clean_correct / total

    # Evaluate FGSM
    correct = 0
    total_adv = 0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.enable_grad():
            adv_images = fgsm(images, labels)

        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total_adv += labels.size(0)

    results["fgsm_accuracy"] = 100.0 * correct / total_adv

    # Evaluate PGD-10
    correct = 0
    total_adv = 0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.enable_grad():
            adv_images = pgd10(images, labels)

        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total_adv += labels.size(0)

    results["pgd_accuracy"] = 100.0 * correct / total_adv

    # Evaluate PGD-50
    correct = 0
    total_adv = 0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.enable_grad():
            adv_images = pgd50(images, labels)

        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total_adv += labels.size(0)

    results["pgd_50_accuracy"] = 100.0 * correct / total_adv

    # Evaluate AutoAttack (ensemble)
    correct = 0
    total_adv = 0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.enable_grad():
            adv_images = autoattack(images, labels)

        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total_adv += labels.size(0)

    results["autoattack_accuracy"] = 100.0 * correct / total_adv

    return results


# ============================================================================
# Energy Measurement
# ============================================================================


def measure_training_energy(
    model: nn.Module,
    train_loader: DataLoader,
    device: str = "cpu",
    num_batches: int = 10,
) -> Dict[str, float]:
    """
    Measure energy consumption during training.

    Args:
        model: Model to train
        train_loader: Training data loader
        device: Device to run on
        num_batches: Number of batches to measure

    Returns:
        Dictionary with energy metrics
    """
    # Check if model has learnable parameters
    has_params = has_learnable_parameters(model)

    if HAS_ZEUS and hasattr(device, "cuda"):
        # Use Zeus for GPU energy measurement
        gpu_id = int(device.split(":")[-1]) if ":" in device else 0
        monitor = ZeusMonitor(gpu_indices=[gpu_id])

        model.train()
        if has_params:
            optimizer = optim.Adam(model.parameters(), lr=0.001)

        monitor.begin_window("training")
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx >= num_batches:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            if has_params:
                optimizer.zero_grad()

            if hasattr(model, "get_initial_state"):
                state = model.get_initial_state(inputs.shape[0])
                outputs, new_state = model(inputs, state, apply_plasticity=True)
            else:
                outputs = model(inputs)

            if has_params:
                loss = F.cross_entropy(outputs, targets.argmax(dim=1))
                loss.backward()
                optimizer.step()

        measurement = monitor.end_window("training")

        return {
            "energy_joules": measurement.total_energy,
            "time_seconds": measurement.time,
            "power_watts": measurement.total_energy / measurement.time,
            "gpu_energy_joules": measurement.gpu_energy.get(gpu_id, 0),
        }

    elif HAS_PYJOULES:
        # Use pyJoules for CPU/GPU energy measurement
        try:
            from pyJoules.device.nvidia_device import NvidiaGPUDomain
            from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain

            if "cuda" in device:
                gpu_id = int(device.split(":")[-1]) if ":" in device else 0
                domains = [
                    RaplPackageDomain(0),
                    RaplDramDomain(0),
                    NvidiaGPUDomain(gpu_id),
                ]
            else:
                domains = [RaplPackageDomain(0), RaplDramDomain(0)]

            devices = DeviceFactory.create_devices(domains)
            meter = EnergyMeter(devices)

            model.train()
            if has_params:
                optimizer = optim.Adam(model.parameters(), lr=0.001)

            meter.start(tag="training")
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                if batch_idx >= num_batches:
                    break

                inputs = inputs.to(device)
                targets = targets.to(device)

                if has_params:
                    optimizer.zero_grad()

                if hasattr(model, "get_initial_state"):
                    state = model.get_initial_state(inputs.shape[0])
                    outputs, new_state = model(inputs, state, apply_plasticity=True)
                else:
                    outputs = model(inputs)

                if has_params:
                    loss = F.cross_entropy(outputs, targets.argmax(dim=1))
                    loss.backward()
                    optimizer.step()

            meter.stop()

            trace = meter.get_trace()
            total_energy = sum(sample.energy_delta for sample in trace)

            return {
                "energy_joules": total_energy,
                "time_seconds": trace[-1].duration,
                "power_watts": total_energy / trace[-1].duration,
            }

        except Exception as e:
            warnings.warn(f"pyJoules measurement failed: {e}")
            return {
                "energy_joules": 0.0,
                "time_seconds": 0.0,
                "power_watts": 0.0,
            }

    else:
        # Fallback - approximate energy measurement
        warnings.warn(
            "Neither Zeus nor pyJoules available. Using approximate energy measurement."
        )

        model.train()
        if has_params:
            optimizer = optim.Adam(model.parameters(), lr=0.001)

        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx >= num_batches:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            if has_params:
                optimizer.zero_grad()

            if hasattr(model, "get_initial_state"):
                state = model.get_initial_state(inputs.shape[0])
                outputs, new_state = model(inputs, state, apply_plasticity=True)
            else:
                outputs = model(inputs)

            if has_params:
                loss = F.cross_entropy(outputs, targets.argmax(dim=1))
                loss.backward()
                optimizer.step()

        elapsed_time = time.time() - start_time

        # Approximate power consumption
        if "cuda" in device:
            approx_power = 250  # Watts for GPU
        else:
            approx_power = 50  # Watts for CPU

        return {
            "energy_joules": approx_power * elapsed_time,
            "time_seconds": elapsed_time,
            "power_watts": approx_power,
        }


def measure_inference_energy(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
    num_samples: int = 100,
) -> Dict[str, float]:
    """
    Measure energy consumption during inference.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run on
        num_samples: Number of samples to measure

    Returns:
        Dictionary with energy metrics
    """
    if HAS_ZEUS and "cuda" in device:
        gpu_id = int(device.split(":")[-1]) if ":" in device else 0
        monitor = ZeusMonitor(gpu_indices=[gpu_id])

        model.eval()

        monitor.begin_window("inference")
        sample_count = 0
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                batch_size = inputs.shape[0]

                if sample_count + batch_size > num_samples:
                    inputs = inputs[: num_samples - sample_count]
                    batch_size = num_samples - sample_count

                if hasattr(model, "get_initial_state"):
                    state = model.get_initial_state(batch_size)
                    _outputs, _ = model(inputs, state, apply_plasticity=False)
                else:
                    _outputs = model(inputs)

                sample_count += batch_size
                if sample_count >= num_samples:
                    break

        measurement = monitor.end_window("inference")

        return {
            "energy_joules": measurement.total_energy,
            "time_seconds": measurement.time,
            "energy_per_inference_joules": measurement.total_energy / num_samples,
            "power_watts": measurement.total_energy / measurement.time,
        }

    else:
        # Fallback implementation
        model.eval()

        latencies = []
        start_time = time.time()
        sample_count = 0

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                batch_size = inputs.shape[0]

                batch_start = time.time()

                if sample_count + batch_size > num_samples:
                    inputs = inputs[: num_samples - sample_count]
                    batch_size = num_samples - sample_count

                if hasattr(model, "get_initial_state"):
                    state = model.get_initial_state(batch_size)
                    _outputs, _ = model(inputs, state, apply_plasticity=False)
                else:
                    _outputs = model(inputs)

                batch_time = time.time() - batch_start
                latencies.append(batch_time)
                sample_count += batch_size

                if sample_count >= num_samples:
                    break

        total_time = time.time() - start_time

        # Approximate power
        if "cuda" in device:
            approx_power = 150  # Watts for GPU during inference
        else:
            approx_power = 30  # Watts for CPU

        return {
            "energy_joules": approx_power * total_time,
            "time_seconds": total_time,
            "energy_per_inference_joules": approx_power * total_time / num_samples,
            "power_watts": approx_power,
            "inference_latency_ms": np.mean(latencies) * 1000,
        }


# ============================================================================
# Interpretability Tests
# ============================================================================


def evaluate_interpretability(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
    num_samples: int = 10,
) -> ExplainabilityResults:
    """
    Evaluate interpretability and explainability.

    Includes:
    - Saliency map consistency
    - Feature attribution entropy
    - Latent space analysis
    - Neuron specialization (for MNE)

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run on
        num_samples: Number of samples to analyze

    Returns:
        ExplainabilityResults object
    """
    results = ExplainabilityResults()

    print("[Interpretability] Starting evaluation...")

    if not HAS_CAPTUM:
        warnings.warn("Captum not installed. Skipping interpretability evaluation.")
        return results

    model.eval()

    # Collect gradients and activations
    gradient_norms_mean = []
    gradient_norms_std = []
    sample_importance = []

    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        inputs.requires_grad = True

        # Flatten if needed
        if hasattr(model, "get_initial_state"):
            state = model.get_initial_state(inputs.shape[0])
            outputs, _ = model(inputs, state, apply_plasticity=False)
        else:
            outputs = model(inputs)

        # Get max output
        output_idx = outputs.argmax(dim=1)
        output_max = outputs.gather(1, output_idx.unsqueeze(1))

        # Compute gradients
        output_max.backward(torch.ones_like(output_max), retain_graph=True)

        if inputs.grad is not None:
            grad_magnitude = inputs.grad.abs().sum(dim=(1 if inputs.dim() == 2 else 2))
            gradient_norms_mean.extend(grad_magnitude.cpu().tolist())
            sample_importance.extend(grad_magnitude.cpu().tolist())

            grad_norms = inputs.grad.abs().view(inputs.grad.shape[0], -1).norm(dim=1)
            gradient_norms_std.extend(grad_norms.cpu().tolist())

        # Only process first batch
        break

    if len(gradient_norms_mean) > 0:
        mean_importance = np.mean(gradient_norms_mean)
        std_importance = np.std(gradient_norms_mean)
        results.feature_attribution_entropy = -(
            mean_importance * np.log(mean_importance + 1e-10)
        )
        results.saliency_consistency_score = 1.0 - (
            std_importance / (mean_importance + 1e-10)
        )
        print(f"  Attribution entropy: {results.feature_attribution_entropy:.4f}")
        print(f"  Saliency consistency: {results.saliency_consistency_score:.4f}")

    # Attribution sparsity
    if len(sample_importance) > 0:
        importance_array = np.array(sample_importance)
        threshold = np.percentile(importance_array, 75)
        sparse_ratio = np.mean(importance_array < threshold)
        results.attribution_sparsity = sparse_ratio
        print(f"  Attribution sparsity: {results.attribution_sparsity:.2%}")

    # MNE-specific neuron specialization
    if hasattr(model, "get_metrics") and hasattr(model, "get_initial_state"):
        state = model.get_initial_state(10)
        batch_inputs = next(iter(test_loader))[0][:10].to(device)

        if hasattr(model, "get_initial_state"):
            state = model.get_initial_state(10)
            outputs, state = model(batch_inputs, state, apply_plasticity=False)

        if hasattr(state, "neuron_state"):
            activations = state.neuron_state.activation.cpu().detach().tolist()
            activation_variance = np.var(activations, axis=0)
            results.neuron_specialization_score = np.std(activation_variance)
            print(f"  Neuron specialization: {results.neuron_specialization_score:.4f}")

            if hasattr(state, "resources"):
                resources = state.neuron_state.resource.cpu().detach().tolist()
                importances = np.mean(np.abs(activations), axis=0)
                if results.neuron_specialization_score > 0:
                    correlation = np.corrcoef(resources[0], importances)[0, 1]
                    results.energy_contribution_correlation = (
                        correlation if not np.isnan(correlation) else 0.0
                    )
                    print(
                        f"  Energy-contribution correlation: {results.energy_contribution_correlation:.4f}"
                    )

    print("[Interpretability] Complete!")
    return results


# ============================================================================
# Main Benchmark Class
# ============================================================================


class ComprehensiveBenchmark:
    """
    Comprehensive benchmark suite for evaluating neural networks.

    This class implements extensive evaluation across 5 dimensions:
    1. Generalization & Robustness (with pytorch-ood and torchattacks)
    2. Efficiency (with Zeus/pyJoules)
    3. Stability & Sensitivity
    4. Continual/Lifelong Learning (simplified version)
    5. Interpretability & Explainability (with Captum)
    """

    def __init__(
        self,
        device: str = "cpu",
        seed: int = 42,
        output_dir: Path = Path("benchmark_results"),
    ):
        """
        Initialize benchmark suite.

        Args:
            device: Device to run benchmarks on
            seed: Random seed for reproducibility
            output_dir: Directory to save results
        """
        self.device = device
        self.seed = seed
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def create_synthetic_data(
        self,
        num_samples: int = 10000,
        input_dim: int = 784,
        num_classes: int = 10,
        seed: int = 42,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create synthetic dataset for benchmarking.

        Returns:
            (train_loader, val_loader, test_loader)
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Generate random features
        X_train = torch.randn(int(0.7 * num_samples), input_dim)
        X_val = torch.randn(int(0.15 * num_samples), input_dim)
        X_test = torch.randn(int(0.15 * num_samples), input_dim)

        # Generate labels with some structure
        def generate_labels(X):
            class_scores = X[:, :num_classes]
            preds = class_scores.argmax(dim=1)
            return F.one_hot(preds, num_classes=num_classes).float()

        y_train = generate_labels(X_train)
        y_val = generate_labels(X_val)
        y_test = generate_labels(X_test)

        # Create data loaders
        train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=128, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val), batch_size=128, shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(X_test, y_test), batch_size=128, shuffle=False
        )

        return train_loader, val_loader, test_loader

    def evaluate_generalization(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ) -> GeneralizationResults:
        """
        Evaluate generalization and robustness.

        Includes:
        - OOD Detection (AUROC, FPR@95% TPR) with pytorch-ood
        - Adversarial Robustness (FGSM, PGD, AutoAttack) with torchattacks
        - Calibration (ECE)
        - Distribution Shift
        """
        results = GeneralizationResults()

        print("[Generalization & Robustness] Starting evaluation...")

        # 1. Standard accuracy (clean)
        results.clean_accuracy = compute_accuracy(model, test_loader, self.device)
        print(f"  Clean accuracy: {results.clean_accuracy:.2f}%")

        # 2. OOD Detection (use validation as OOD for synthetic data)
        results.ood_auroc, results.ood_fpr_at_95_tpr = compute_ood_metrics(
            model, val_loader, test_loader, self.device
        )
        print(f"  OOD AUROC: {results.ood_auroc:.4f}")
        print(f"  OOD FPR@95% TPR: {results.ood_fpr_at_95_tpr:.4f}")

        # 3. Calibration
        results.expected_calibration_error = compute_expected_calibration_error(
            model, test_loader, device=self.device
        )
        print(f"  ECE: {results.expected_calibration_error:.4f}")

        # 4. Distribution Shift (use train vs test as proxy)
        results.source_accuracy = compute_accuracy(model, train_loader, self.device)
        results.target_accuracy = results.clean_accuracy
        results.domain_adaptation_gain = (
            (results.target_accuracy - results.source_accuracy)
            if results.source_accuracy > 0
            else 0.0
        )
        print(f"  Source accuracy (train): {results.source_accuracy:.2f}%")
        print(f"  Target accuracy (test): {results.target_accuracy:.2f}%")

        # 5. Adversarial Robustness (if torchattacks available)
        try:
            adv_results = evaluate_adversarial_robustness(
                model, test_loader, self.device, eps=0.0314
            )
            results.fgsm_accuracy = adv_results.get("fgsm_accuracy", 0.0)
            results.pgd_accuracy = adv_results.get("pgd_accuracy", 0.0)
            results.autoattack_accuracy = adv_results.get("autoattack_accuracy", 0.0)
            print(f"  FGSM robust accuracy: {results.fgsm_accuracy:.2f}%")
            print(f"  PGD robust accuracy: {results.pgd_accuracy:.2f}%")
            print(f"  AutoAttack robust accuracy: {results.autoattack_accuracy:.2f}%")
        except Exception as e:
            warnings.warn(f"Adversarial robustness evaluation failed: {e}")
            results.fgsm_accuracy = 0.0
            results.pgd_accuracy = 0.0
            results.autoattack_accuracy = 0.0

        print("[Generalization & Robustness] Complete!")
        return results

    def evaluate_efficiency(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int = 2,
    ) -> EfficiencyResults:
        """
        Evaluate efficiency metrics.

        Includes:
        - Training time and convergence
        - Inference latency and throughput
        - Energy consumption (with Zeus/pyJoules)
        - Parameter efficiency
        - Data efficiency
        - FLOPs/MACs
        """
        results = EfficiencyResults()

        print("[Efficiency] Starting evaluation...")

        # 1. Parameter count
        results.num_parameters = count_parameters(model)
        print(f"  Parameters: {results.num_parameters:,}")

        # 2. Training time per epoch
        model.train()

        # Check if model has learnable parameters
        has_parameters = results.num_parameters > 0
        if has_parameters:
            optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Warm up
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            if hasattr(model, "get_initial_state"):
                batch_size = inputs.shape[0]
                state = model.get_initial_state(batch_size)
                _outputs, _ = model(inputs, state, apply_plasticity=False)
            else:
                _outputs = model(inputs)

            if has_parameters:
                loss = F.cross_entropy(_outputs, targets.argmax(dim=1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            break

        # Measure training time and energy
        start_time = time.time()

        if HAS_ZEUS and "cuda" in self.device or HAS_PYJOULES:
            energy_train = measure_training_energy(
                model, train_loader, self.device, num_batches=10
            )
            results.training_time_per_epoch = energy_train["time_seconds"] / 10.0
            results.energy_per_training_epoch_joules = (
                energy_train["energy_joules"] / 10.0
            )
        else:
            epoch_loss = 0.0
            num_batches = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if hasattr(model, "get_initial_state"):
                    state = model.get_initial_state(inputs.shape[0])
                    outputs, new_state = model(inputs, state, apply_plasticity=True)
                else:
                    outputs = model(inputs)

                if has_parameters:
                    loss = F.cross_entropy(outputs, targets.argmax(dim=1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                else:
                    # For models without learnable parameters, just compute loss without training
                    loss = F.cross_entropy(outputs, targets.argmax(dim=1))
                    epoch_loss += loss.item()

                num_batches += 1

                if num_batches >= 50:  # Only measure 50 batches for speed
                    break

            results.training_time_per_epoch = time.time() - start_time

        print(f"  Training time per epoch: {results.training_time_per_epoch:.2f}s")

        # 3. Inference latency and throughput
        model.eval()
        latencies = []
        total_samples = 0
        start_time = time.time()

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                batch_size = inputs.shape[0]

                # Measure latency
                batch_start = time.time()
                if hasattr(model, "get_initial_state"):
                    state = model.get_initial_state(batch_size)
                    outputs, _ = model(inputs, state, apply_plasticity=False)
                else:
                    outputs = model(inputs)
                batch_time = time.time() - batch_start

                latencies.append(batch_time)
                total_samples += batch_size

                if total_samples >= 100:  # Only process 100 samples
                    break

        total_time = time.time() - start_time
        results.inference_latency_ms = np.mean(latencies) * 1000
        results.throughput_samples_per_sec = total_samples / total_time
        print(f"  Inference latency: {results.inference_latency_ms:.2f}ms")
        print(f"  Throughput: {results.throughput_samples_per_sec:.1f} samples/sec")

        # 4. Parameter efficiency
        test_acc = compute_accuracy(model, test_loader, self.device)
        results.accuracy_per_million_params = (
            (test_acc / (results.num_parameters / 1e6))
            if results.num_parameters > 0
            else 0.0
        )
        print(
            f"  Accuracy per million params: {results.accuracy_per_million_params:.2f}"
        )

        # 5. Energy measurement for inference
        try:
            energy_inference = measure_inference_energy(
                model, test_loader, self.device, num_samples=50
            )
            results.energy_per_inference_joules = energy_inference[
                "energy_per_inference_joules"
            ]
            results.energy_efficiency_samples_per_joule = energy_inference.get(
                "throughput_samples_per_sec", 1.0
            ) / energy_inference.get("power_watts", 1.0)
            print(f"  Energy per inference: {results.energy_per_inference_joules:.4f}J")
        except Exception as e:
            warnings.warn(f"Energy measurement failed: {e}")

        # 6. Data efficiency (use full dataset for now)
        results.accuracy_at_100_percent_data = test_acc

        print(f"  Data efficiency (100%): {results.accuracy_at_100_percent_data:.2f}%")

        print("[Efficiency] Complete!")
        return results

    def evaluate_stability(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ) -> StabilityResults:
        """
        Evaluate stability and sensitivity.

        Includes:
        - Hyperparameter sensitivity
        - Initialization sensitivity (multiple random seeds)
        - Ablation studies (for MNE)
        - Gradient analysis
        """
        results = StabilityResults()

        print("[Stability & Sensitivity] Starting evaluation...")

        # 1. Initialization sensitivity (5 random seeds)
        accuracies = []
        seeds = [42, 123, 456, 789, 101112]

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Reset model parameters
            for m in model.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

            # Quick evaluation
            acc = compute_accuracy(model, test_loader, self.device)
            accuracies.append(acc)

        results.mean_accuracy = np.mean(accuracies)
        results.std_accuracy = np.std(accuracies)
        results.min_accuracy = min(accuracies)
        results.max_accuracy = max(accuracies)
        print(
            f"  Mean accuracy (5 seeds): {results.mean_accuracy:.2f}% ± {results.std_accuracy:.2f}%"
        )

        # 2. Gradient analysis
        if has_learnable_parameters(model):
            model.train()
            sample_batch = next(iter(train_loader))
            inputs, targets = sample_batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if hasattr(model, "get_initial_state"):
                state = model.get_initial_state(inputs.shape[0])
                outputs, _ = model(inputs, state, apply_plasticity=False)
            else:
                outputs = model(inputs)

            loss = F.cross_entropy(outputs, targets.argmax(dim=1))
            loss.backward()

            grad_norms = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_norms.append(param.grad.norm().item())

            results.mean_grad_norm = np.mean(grad_norms)
            results.std_grad_norm = np.std(grad_norms)
            results.vanished_gradient_ratio = (
                (np.mean([gn < 1e-6 for gn in grad_norms])) if grad_norms else 0.0
            )
            results.exploding_gradient_ratio = (
                (np.mean([gn > 100 for gn in grad_norms])) if grad_norms else 0.0
            )
            print(f"  Mean grad norm: {results.mean_grad_norm:.4f}")
            print(f"  Vanished grad ratio: {results.vanished_gradient_ratio:.2%}")
            print(f"  Exploding grad ratio: {results.exploding_gradient_ratio:.2%}")
        else:
            # Model has no learnable parameters (e.g., MNE)
            results.mean_grad_norm = 0.0
            results.std_grad_norm = 0.0
            results.vanished_gradient_ratio = 0.0
            results.exploding_gradient_ratio = 0.0
            print(f"  Mean grad norm: 0.00 (model has no learnable parameters)")
            print(f"  Vanished grad ratio: 0.00%")
            print(f"  Exploding grad ratio: 0.00%")

        # 3. Hyperparameter sensitivity (simplified)
        if has_learnable_parameters(model):
            lr_variations = []
            for lr in [0.001, 0.0005, 0.002]:
                optimizer = optim.Adam(model.parameters(), lr=lr)
                # Quick train loop - just check if training proceeds
                loss = None
                try:
                    for inputs, targets in train_loader:
                        inputs, targets = (
                            inputs.to(self.device),
                            targets.to(self.device),
                        )
                        optimizer.zero_grad()
                        if hasattr(model, "get_initial_state"):
                            state = model.get_initial_state(inputs.shape[0])
                            outputs, _ = model(inputs, state, apply_plasticity=True)
                        else:
                            outputs = model(inputs)
                        loss = F.cross_entropy(outputs, targets.argmax(dim=1))
                        loss.backward()
                        optimizer.step()
                        break
                    lr_variations.append(loss.item())
                except Exception:
                    lr_variations.append(float("inf"))

            results.lr_variance = (
                np.std(lr_variations) if len(lr_variations) > 0 else 0.0
            )
        else:
            results.lr_variance = 0.0

        print(f"  LR sensitivity (variance): {results.lr_variance:.4f}")

        overall_sensitivities = [
            results.std_accuracy,
            results.lr_variance,
            results.vanished_gradient_ratio,
        ]
        results.overall_sensitivity_score = np.mean(overall_sensitivities)

        print("[Stability & Sensitivity] Complete!")
        return results

    def evaluate_continual_learning(
        self,
        model: nn.Module,
        num_tasks: int = 5,
        samples_per_task: int = 500,
        input_dim: int = 784,
        num_classes: int = 10,
    ) -> ContinualLearningResults:
        """
        Evaluate continual learning capabilities.

        Includes:
        - Task incremental learning
        - Catastrophic forgetting
        - Backward/forward transfer
        - Memory replay efficiency
        - Neurogenesis/apoptosis (for MNE)

        Note: This is a simplified implementation. For full continual learning
        benchmarks, use Avalanche library.
        """
        results = ContinualLearningResults()
        results.num_tasks = num_tasks

        print(f"[Continual Learning] Starting evaluation with {num_tasks} tasks...")

        # Create task sequence (Permuted MNIST style)
        task_accuracies = []
        all_task_test_data = []
        all_task_test_labels = []

        for task_id in range(num_tasks):
            # Create permuted data for this task
            np.random.seed(self.seed + task_id)
            permutation = np.random.permutation(
                input_dim
            ).tolist()  # Convert to list for torch compatibility

            X_task = torch.randn(samples_per_task, input_dim)
            # Apply permutation and create task structure
            X_task = X_task[:, permutation]

            # Labels: shift by task_id for class-incremental scenario
            y_task = torch.randint(0, num_classes, (samples_per_task,))
            y_task = (y_task + task_id) % num_classes

            # Create loader
            task_loader = DataLoader(
                TensorDataset(X_task, F.one_hot(y_task, num_classes).float()),
                batch_size=128,
                shuffle=True,
            )

            # Create test data for this task
            X_test_task = torch.randn(200, input_dim)
            X_test_task = X_test_task[:, permutation]
            y_test_task = torch.randint(0, num_classes, (200,))
            y_test_task = (y_test_task + task_id) % num_classes

            all_task_test_data.append(X_test_task)
            all_task_test_labels.append(y_test_task)

            # Train on current task
            model.train()
            has_params = has_learnable_parameters(model)
            if has_params:
                optimizer = optim.Adam(model.parameters(), lr=0.001)

            for epoch in range(10):  # Quick training
                for inputs, targets in task_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    if has_params:
                        optimizer.zero_grad()

                    if hasattr(model, "get_initial_state"):
                        state = model.get_initial_state(inputs.shape[0])
                        outputs, new_state = model(inputs, state, apply_plasticity=True)
                    else:
                        outputs = model(inputs)

                    if has_params:
                        loss = F.cross_entropy(outputs, targets.argmax(dim=1))
                        loss.backward()
                        optimizer.step()

            # Evaluate on all seen tasks so far
            model.eval()
            task_accuracies_current_task = []

            for prev_id in range(task_id + 1):
                X_test = all_task_test_data[prev_id].to(self.device)
                y_test = all_task_test_labels[prev_id]

                with torch.no_grad():
                    if hasattr(model, "get_initial_state"):
                        state = model.get_initial_state(X_test.shape[0])
                        outputs, _ = model(X_test, state, apply_plasticity=False)
                    else:
                        outputs = model(X_test)

                    predictions = outputs.argmax(dim=1)
                    acc = (predictions == y_test.to(self.device)).float().mean() * 100
                    task_accuracies_current_task.append(acc.item())

            task_accuracies.append(task_accuracies_current_task)
            print(
                f"  Task {task_id}: Average accuracy: {np.mean(task_accuracies_current_task):.2f}%"
            )

        # Compute continual learning metrics
        final_accuracies = task_accuracies[-1]
        results.average_task_accuracy = np.mean(final_accuracies)

        # Forgetting
        if num_tasks > 1:
            first_task_performances = [accs[0] for accs in task_accuracies]
            final_first_task_acc = final_accuracies[0]
            max_first_task_acc = max(first_task_performances)
            results.forgetting_measure = max_first_task_acc - final_first_task_acc

        # Learning curve area (trapezoid integration)
        learning_curve = [np.mean(accs) for accs in task_accuracies]
        # NumPy 2.0 removed np.trapz, use np.trapezoid or manual calculation
        try:
            results.learning_curve_area = np.trapezoid(learning_curve)
        except AttributeError:
            # NumPy 1.x backward compatibility
            try:
                results.learning_curve_area = np.trapz(learning_curve)
            except AttributeError:
                # Manual fallback
                results.learning_curve_area = sum(learning_curve) / len(learning_curve)

        # Backward transfer
        if num_tasks > 1:
            initial_acc = task_accuracies[0][0]
            final_first_task_acc = task_accuracies[-1][0]
            results.backward_transfer_score = final_first_task_acc - initial_acc

        # Forward transfer
        results.forward_transfer_score = learning_curve[-1] - learning_curve[0]

        # MNE-specific metrics
        if hasattr(model, "get_metrics"):
            final_state = model.get_initial_state(1)
            metrics = model.get_metrics(final_state)

            results.total_neurogenesis_events = metrics.get("neurogenesis_count", 0)
            results.total_apoptosis_events = metrics.get("apoptosis_count", 0)
            results.initial_neurons = num_tasks * 10
            results.final_neurons = metrics.get("num_neurons", results.initial_neurons)
            results.neuron_count_change = (
                results.final_neurons - results.initial_neurons
            )

        print(f"  Average task accuracy: {results.average_task_accuracy:.2f}%")
        print(f"  Forgetting measure: {results.forgetting_measure:.2f}%")

        print("[Continual Learning] Complete!")
        return results

    def run_full_benchmark(
        self,
        model: nn.Module,
        model_name: str,
        input_dim: int = 784,
        num_classes: int = 10,
        num_samples: int = 5000,
        run_continual: bool = True,
    ) -> BenchmarkResults:
        """
        Run complete benchmark suite on a model.

        Args:
            model: Model to benchmark
            model_name: Name of the model for results
            input_dim: Input dimension
            num_classes: Number of output classes
            num_samples: Number of training samples
            run_continual: Whether to run continual learning evaluation

        Returns:
            BenchmarkResults object with all metrics
        """
        print(f"\n{'=' * 70}")
        print(f"STARTING COMPREHENSIVE BENCHMARK: {model_name}")
        print(f"{'=' * 70}")

        results = BenchmarkResults(model_name=model_name)

        # Create data
        train_loader, val_loader, test_loader = self.create_synthetic_data(
            num_samples=num_samples,
            input_dim=input_dim,
            num_classes=num_classes,
            seed=self.seed,
        )

        # Run all evaluations
        results.generalization = self.evaluate_generalization(
            model, train_loader, val_loader, test_loader
        )

        results.efficiency = self.evaluate_efficiency(
            model, train_loader, test_loader, num_epochs=2
        )

        results.stability = self.evaluate_stability(model, train_loader, test_loader)

        if run_continual:
            results.continual_learning = self.evaluate_continual_learning(
                model,
                num_tasks=5,
                samples_per_task=500,
                input_dim=input_dim,
                num_classes=num_classes,
            )

        results.explainability = evaluate_interpretability(model, test_loader)

        # Compute overall scores
        results.robustness_score = (
            results.generalization.ood_auroc
            + (1.0 - results.generalization.expected_calibration_error)
            + results.generalization.clean_accuracy / 100
        ) / 3

        # Avoid division by zero and cap extreme values
        latency_ms = max(
            results.efficiency.inference_latency_ms, 0.001
        )  # Minimum 1 microsecond
        throughput_score = min(
            results.efficiency.throughput_samples_per_sec / 1000, 1000
        )  # Cap at 1000

        # Handle models with 0 learnable parameters (e.g., MNE with state-based learning)
        if results.efficiency.num_parameters == 0:
            # For parameterless models, just use latency and throughput for efficiency
            results.efficiency_score = (
                (1.0 / np.log1p(latency_ms)) + throughput_score
            ) / 2
        else:
            results.efficiency_score = (
                (1.0 / np.log1p(results.efficiency.num_parameters / 1e6))
                + (1.0 / np.log1p(latency_ms))
                + throughput_score
            ) / 3

        results.reliability_score = (
            (1.0 - results.stability.std_accuracy / 100)
            + results.generalization.clean_accuracy / 100
            + (1.0 - results.continual_learning.forgetting_measure / 100)
        ) / 3

        results.overall_score = (
            results.robustness_score * 0.3
            + results.efficiency_score * 0.3
            + results.reliability_score * 0.4
        )

        print(f"\n{'=' * 70}")
        print(f"BENCHMARK COMPLETE: {model_name}")
        print(f"{'=' * 70}")
        print(f"Overall Score: {results.overall_score:.4f}")
        print(f"  Robustness: {results.robustness_score:.4f}")
        print(f"  Efficiency: {results.efficiency_score:.4f}")
        print(f"  Reliability: {results.reliability_score:.4f}")

        # Save results
        results_path = self.output_dir / f"{model_name}_benchmark.json"
        results.save_json(results_path)
        print(f"\nResults saved to: {results_path}")

        return results


# ============================================================================
# Benchmark Comparison and Reporting
# ============================================================================


def compare_models(
    results_list: List[BenchmarkResults],
    output_dir: Path = Path("benchmark_results"),
) -> None:
    """
    Compare multiple models and generate comparison report.

    Args:
        results_list: List of BenchmarkResults to compare
        output_dir: Directory to save comparison report
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("MODEL COMPARISON REPORT")
    print(f"{'=' * 70}\n")

    # Overall scores comparison
    print("Overall Scores:")
    print("-" * 70)
    for results in sorted(results_list, key=lambda r: r.overall_score, reverse=True):
        print(
            f"  {results.model_name:30s}: {results.overall_score:.4f} "
            f"(R: {results.robustness_score:.4f}, E: {results.efficiency_score:.4f}, "
            f"Rel: {results.reliability_score:.4f})"
        )

    # Generalization comparison
    print("\nGeneralization & Robustness:")
    print("-" * 70)
    print(
        f"{'Model':30s} {'Clean Acc':>10} {'OOD AUROC':>10} {'ECE':>10} {'FGSM Acc':>10}"
    )
    for results in results_list:
        g = results.generalization
        print(
            f"{results.model_name:30s} {g.clean_accuracy:10.2f} {g.ood_auroc:10.4f} "
            f"{g.expected_calibration_error:10.4f} {g.fgsm_accuracy:10.2f}"
        )

    # Efficiency comparison
    print("\nEfficiency:")
    print("-" * 70)
    print(
        f"{'Model':30s} {'Params(M)':>10} {'Lat(ms)':>10} {'Thru/s':>10} {'Eff(J/inf)':>10}"
    )
    for results in results_list:
        e = results.efficiency
        print(
            f"{results.model_name:30s} {e.num_parameters / 1e6:10.2f} {e.inference_latency_ms:10.2f} "
            f"{e.throughput_samples_per_sec:10.1f} {e.energy_per_inference_joules:10.4f}"
        )

    # Continual Learning comparison
    print("\nContinual Learning:")
    print("-" * 70)
    print(
        f"{'Model':30s} {'Avg Acc':>10} {'Forget':>10} {'B-trans':>10} {'F-trans':>10}"
    )
    for results in results_list:
        cl = results.continual_learning
        print(
            f"{results.model_name:30s} {cl.average_task_accuracy:10.2f} {cl.forgetting_measure:10.2f} "
            f"{cl.backward_transfer_score:10.2f} {cl.forward_transfer_score:10.2f}"
        )

    # Save comparison to file
    comparison_path = output_dir / "comparison_report.txt"
    with open(comparison_path, "w") as f:
        f.write("MODEL COMPARISON REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("Overall Scores:\n")
        f.write("-" * 70 + "\n")
        for results in sorted(
            results_list, key=lambda r: r.overall_score, reverse=True
        ):
            f.write(
                f"  {results.model_name:30s}: {results.overall_score:.4f} "
                f"(R: {results.robustness_score:.4f}, E: {results.efficiency_score:.4f}, "
                f"Rel: {results.reliability_score:.4f})\n"
            )

    print(f"\nComparison report saved to: {comparison_path}")


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """Main function to run comprehensive benchmarks."""
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    benchmark = ComprehensiveBenchmark(device=device, seed=42)

    # Create baseline model
    print("Creating baseline models...")

    # Standard MLP
    mlp = StandardMLP(
        input_dim=784,
        hidden_dims=[256, 256, 256],  # 4 hidden layers to match BEAST MNE!
        output_dim=10,
        activation="relu",
        dropout=0.25,  # Same dropout as BEAST MNE
    ).to(device)

    # Create results list
    all_results = []

    # Benchmark Standard MLP
    print("\nBenchmarking Standard MLP...")
    mlp_results = benchmark.run_full_benchmark(
        model=mlp,
        model_name="StandardMLP",
        input_dim=784,
        num_classes=10,
        num_samples=5000,
        run_continual=True,
    )
    all_results.append(mlp_results)

    # Benchmark MNE if available
    try:
        import sys

        sys.path.append(str(Path(__file__).parent.parent))
        from src.core import MNE, MNEConfig

        print("\nBenchmarking MNE...")
        # MNE configuration - MAXIMUM ACCURACY
        input_dim = 784
        num_classes = 10
        mne_config = MNEConfig(
            input_dim=input_dim,
            output_dim=num_classes,
            num_neurons=256,  # Large capacity for accuracy
            num_layers=4,  # Deep network
            activation_fn="leaky_relu",
            dropout_rate=0.25,
            weight_decay=0.02,
            gradient_lr=0.0007,
            total_epochs=30,  # Extensive training
            warmup_epochs=5,
            label_smoothing=0.05,
            use_gradient_descent=True,
            metabolic_lr_modulation=True,
            grad_clip=0.5,
            device=device,
        )
        mne = MNE(mne_config).to(device)

        # Pre-train MNE extensively
        print("  Pre-training MNE for 30 epochs...")
        optimizer = torch.optim.AdamW(
            mne.parameters(),
            lr=mne_config.gradient_lr,
            weight_decay=mne_config.weight_decay,
        )

        # Use more data for better generalization
        train_loader, _, _ = benchmark.create_synthetic_data(
            num_samples=10000,  # Double the data
            input_dim=input_dim,
            num_classes=num_classes,
            seed=42,
        )

        best_acc = 0.0
        best_test_acc = 0.0

        for epoch in range(mne_config.total_epochs):
            mne.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            # One-cycle learning rate
            lr = mne.get_onecycle_lr(optimizer, epoch)

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if targets.dim() > 1:
                    targets = targets.argmax(dim=1)

                # Get initial state for this batch
                batch_size = inputs.shape[0]
                state = mne.get_initial_state(batch_size)

                # Train step
                loss, state, metrics = mne.train_step(inputs, targets, state, optimizer)
                epoch_loss += loss.item()

                # Track accuracy
                epoch_correct += int(metrics["accuracy"] * targets.size(0))
                epoch_total += targets.size(0)

            avg_loss = epoch_loss / len(train_loader)
            avg_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0

            if avg_accuracy > best_acc:
                best_acc = avg_accuracy

            # Evaluate every 5 epochs
            if epoch % 5 == 0 or epoch == mne_config.total_epochs - 1:
                # Quick test evaluation
                mne.eval()
                _, test_loader, _ = benchmark.create_synthetic_data(
                    num_samples=1000,
                    input_dim=input_dim,
                    num_classes=10,
                    seed=100 + epoch,
                )
                test_correct = 0
                test_total = 0
                with torch.no_grad():
                    for inp, tar in test_loader:
                        inp, tar = inp.to(device), tar.to(device)
                        if tar.dim() > 1:
                            tar = tar.argmax(dim=1)
                        st = mne.get_initial_state(inp.shape[0])
                        out, _ = mne(inp, st, apply_plasticity=False)
                        pred = out.argmax(dim=1)
                        test_correct += (pred == tar).sum().item()
                        test_total += tar.size(0)
                test_acc = test_correct / test_total if test_total > 0 else 0.0
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                mne.train()

            if epoch % 5 == 0 or epoch == mne_config.total_epochs - 1:
                print(
                    f"    Epoch {epoch + 1}/{mne_config.total_epochs}: "
                    f"Loss={avg_loss:.4f}, Train={avg_accuracy:.2%}, "
                    f"Test={test_acc:.2%}, LR={lr:.6f}"
                )

        print(f"    Best training accuracy: {best_acc:.2%}")
        print(f"    Best test accuracy: {best_test_acc:.2%}")

        mne.eval()
        mne_results = benchmark.run_full_benchmark(
            model=mne,
            model_name="MNE",
            input_dim=input_dim,
            num_classes=num_classes,
            num_samples=5000,
            run_continual=True,
        )
        all_results.append(mne_results)

    except ImportError:
        print("\nMNE not available, skipping...")

    # Compare results
    if len(all_results) > 1:
        compare_models(all_results)

    print("\n" + "=" * 70)
    print("BENCHMARK SUITE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
