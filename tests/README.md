# Comprehensive Benchmark Suite for MNE

This directory contains a comprehensive benchmark suite for evaluating the Metabolic Neural Ecosystem (MNE) against conventional neural networks.

## Overview

The benchmark evaluates across **10 dimensions**:

1. **Generalization & Robustness**
   - OOD Detection (AUROC, FPR@95% TPR) with `pytorch-ood`
   - Adversarial Robustness (FGSM, PGD, AutoAttack) with `torchattacks`
   - Calibration (Expected Calibration Error)
   - Distribution Shift
   - Corruption Robustness

2. **Efficiency**
   - Training time / convergence speed
   - Inference throughput and latency
   - Energy consumption with `Zeus`/`pyJoules`
   - Parameter efficiency
   - Data efficiency (1%, 10%, 50%, 100% of data)
   - FLOPs/MACs

3. **Stability & Sensitivity**
   - Hyperparameter sensitivity
   - Initialization sensitivity (multiple seeds)
   - Ablation studies
   - Gradient analysis

4. **Continual/Lifelong Learning**
   - Task incremental learning
   - Catastrophic forgetting
   - Backward/forward transfer
   - Memory replay efficiency
   - MNE-specific: Neurogenesis/Apoptosis tracking

5. **Interpretability & Explainability**
   - Saliency maps (Captum)
   - Feature attribution entropy
   - Latent space analysis
   - Concept attribution (TCAV)
   - Counterfactual explanations

## Installation

Install required external libraries:

```bash
# Core dependencies
pip install torch numpy

# Adversarial robustness
pip install torchattacks

# OOD detection and calibration
pip install pytorch-ood

# Interpretability
pip install captum

# Energy measurement
pip install zeus-ml      # GPU energy measurement
pip install pyJoules       # CPU/GPU energy measurement

# Continual learning (optional)
pip install avalanche-lib

# Data handling
pip install scikit-learn matplotlib
```

**Note on NumPy Compatibility**: If you see a warning about NumPy 2.x vs NumPy 1.x, the benchmark is designed to work with both versions. The code uses `.tolist()` instead of `.numpy()` for tensor conversions to avoid compatibility issues.

## Usage

### Basic Usage

```python
import torch
from tests.benchmark import ComprehensiveBenchmark
from tests.benchmark import StandardMLP
from src.core import MNE, MNEConfig

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
benchmark = ComprehensiveBenchmark(device=device, seed=42)

# Create and benchmark a Standard MLP
mlp = StandardMLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)
mlp_results = benchmark.run_full_benchmark(
    model=mlp,
    model_name="StandardMLP",
    input_dim=784,
    num_classes=10,
    num_samples=5000,
    run_continual=True
)

# Create and benchmark MNE
mne_config = MNEConfig(num_neurons=128, activation_fn="tanh")
mne = MNE(mne_config)
mne_results = benchmark.run_full_benchmark(
    model=mne,
    model_name="MNE",
    input_dim=784,
    num_classes=10,
    num_samples=5000,
    run_continual=True
)

# Compare results
from tests.benchmark import compare_models
compare_models([mlp_results, mne_results])
```

### Advanced Usage

#### Custom Dataset

```python
from torch.utils.data import DataLoader, TensorDataset
import torch

# Create your own dataset
X_train = torch.randn(1000, 784)
y_train = torch.randint(0, 10, (1000,))

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)

# Run benchmark with custom data
results = benchmark.run_full_benchmark(
    model=mne_model,
    model_name="MNE_custom",
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader
)
```

#### Selective Evaluation

```python
# Only run generalization tests
results = benchmark.evaluate_generalization(mne, train_loader, val_loader, test_loader)

# Only run efficiency tests
results_eff = benchmark.evaluate_efficiency(mne, train_loader, test_loader, num_epochs=5)

# Only run interpretability tests
results_exp = benchmark.evaluate_explainability(mne, test_loader)
```

## Output Format

### JSON Results

Each benchmark run saves a JSON file with all metrics:

```json
{
  "model_name": "MNE",
  "generalization": {
    "ood_auroc": 0.85,
    "ood_fpr_at_95_tpr": 0.12,
    "clean_accuracy": 92.5,
    "fgsm_accuracy": 65.2,
    "pgd_accuracy": 58.3,
    "autoattack_accuracy": 52.7,
    "expected_calibration_error": 0.03,
    "brier_score": 0.15,
    "source_accuracy": 91.2,
    "target_accuracy": 92.5,
    "domain_adaptation_gain": 1.3
  },
  "efficiency": {
    "training_time_per_epoch": 2.5,
    "epochs_to_target_accuracy": 15,
    "convergence_rate": 0.8,
    "inference_latency_ms": 5.2,
    "throughput_samples_per_sec": 192.3,
    "energy_per_inference_joules": 0.008,
    "energy_per_training_epoch_joules": 25.5,
    "energy_efficiency_samples_per_joule": 24037.5,
    "num_parameters": 128000,
    "accuracy_per_million_params": 0.722,
    "accuracy_at_1_percent_data": 45.2,
    "accuracy_at_10_percent_data": 78.5,
    "accuracy_at_50_percent_data": 89.1,
    "accuracy_at_100_percent_data": 92.5,
    "flops_per_forward": 12544000,
    "macs_per_forward": 12544000
  },
  "stability": {
    "lr_variance": 2.3,
    "batch_size_variance": 1.8,
    "regularization_variance": 0.5,
    "overall_sensitivity_score": 0.75,
    "mean_accuracy": 90.8,
    "std_accuracy": 1.5,
    "min_accuracy": 88.2,
    "max_accuracy": 93.1,
    "full_model_accuracy": 92.5,
    "no_plasticity_accuracy": 85.3,
    "no_homeostasis_accuracy": 87.9,
    "no_topology_accuracy": 88.7,
    "mean_grad_norm": 0.045,
    "std_grad_norm": 0.012,
    "vanished_gradient_ratio": 0.02,
    "exploding_gradient_ratio": 0.01
  },
  "continual_learning": {
    "num_tasks": 5,
    "average_task_accuracy": 88.5,
    "forgetting_measure": 5.2,
    "learning_curve_area": 425.3,
    "backward_transfer_score": 2.3,
    "knowledge_retention_score": 95.1,
    "forward_transfer_score": 8.7,
    "knowledge_reuse_score": 7.9,
    "replay_buffer_accuracy": 89.2,
    "no_replay_accuracy": 82.1,
    "replay_efficiency_gain": 7.1,
    "total_neurogenesis_events": 45,
    "total_apoptosis_events": 23,
    "initial_neurons": 50,
    "final_neurons": 72,
    "neuron_count_change": 22
  },
  "explainability": {
    "saliency_consistency_score": 0.78,
    "feature_attribution_entropy": 2.45,
    "attribution_sparsity": 0.72,
    "concept_alignment_score": 0.65,
    "concept_diversity_score": 0.73,
    "counterfactual_distance": 0.35,
    "counterfactual_success_rate": 0.85,
    "latent_space_clusters": 10,
    "intra_cluster_distance": 0.23,
    "inter_cluster_distance": 1.85,
    "separation_score": 0.88,
    "neuron_specialization_score": 0.42,
    "energy_contribution_correlation": 0.67
  },
  "overall_score": 0.75,
  "robustness_score": 0.82,
  "efficiency_score": 0.68,
  "reliability_score": 0.74
}
```

### Comparison Report

```
======================================================================
MODEL COMPARISON REPORT
======================================================================

Overall Scores:
----------------------------------------------------------------------
  MNE                          : 0.7500 (R: 0.8200, E: 0.6800, Rel: 0.7400)
  StandardMLP                    : 0.6850 (R: 0.7100, E: 0.7500, Rel: 0.6000)

Generalization & Robustness:
----------------------------------------------------------------------
  Model                          Clean Acc     OOD AUROC    ECE         FGSM Acc
  MNE                          92.50      0.8500      0.0300      65.20
  StandardMLP                    88.30      0.7200      0.0450      42.50

Efficiency:
----------------------------------------------------------------------
  Model                          Params(M)    Lat(ms)      Thru/s       Eff(J/inf)
  MNE                          0.13       5.20        192.3       0.0080
  StandardMLP                    0.35       8.75        125.4       0.0150

Continual Learning:
----------------------------------------------------------------------
  Model                          Avg Acc      Forget      B-trans      F-trans
  MNE                          88.50      5.20        2.30        8.70
  StandardMLP                    75.80      12.30       -3.50       5.40
```

## External Libraries Used

| Library | Purpose | Installation |
|---------|---------|-------------|
| **torchattacks** | Adversarial robustness (FGSM, PGD, AutoAttack) | `pip install torchattacks` |
| **pytorch-od** | OOD detection metrics (AUROC, FPR@95% TPR) | `pip install pytorch-ood` |
| **Captum** | Model interpretability (saliency, GradCAM, IntegratedGradients) | `pip install captum` |
| **Zeus** | GPU energy measurement | `pip install zeus-ml` |
| **PyJoules** | CPU/GPU energy measurement | `pip install pyJoules` |
| **Avalanche** (optional) | Continual learning benchmarks | `pip install avalanche-lib` |

## Key Metrics Explained

### Generalization & Robustness

- **OOD AUROC**: Area under ROC curve for out-of-distribution detection. Higher is better.
- **FPR@95% TPR**: False positive rate at 95% true positive rate. Lower is better.
- **ECE**: Expected Calibration Error. Measures how well model confidence matches accuracy. Lower is better.
- **FGSM/PGD/AutoAttack Robustness**: Accuracy under adversarial attacks. Higher is better.

### Efficiency

- **Training time per epoch**: Time to train one epoch. Lower is better.
- **Inference latency**: Time per single inference. Lower is better.
- **Throughput**: Samples processed per second. Higher is better.
- **Energy per inference**: Joules per single prediction. Lower is better.
- **Parameters**: Total model parameters. Depends on model complexity.
- **Accuracy per million params**: Accuracy normalized by param count. Higher is better.

### Stability

- **Mean accuracy (5 seeds)**: Average across 5 random initializations.
- **Std accuracy**: Standard deviation across seeds. Lower is more stable.
- **Vanished gradient ratio**: Proportion of vanished gradients. Should be low.
- **Exploding gradient ratio**: Proportion of exploding gradients. Should be low.

### Continual Learning

- **Average task accuracy**: Average accuracy across all tasks.
- **Forgetting measure**: How much performance drops on earlier tasks. Lower is better.
- **Backward transfer**: Does learning new tasks improve old ones? Positive is better.
- **Forward transfer**: Does prior knowledge help learn new tasks? Positive is better.
- **Neurogenesis/Apoptosis**: MNE-specific metrics.

### Interpretability

- **Saliency consistency**: Consistency of gradient-based explanations. Higher is better.
- **Attribution sparsity**: How sparse the feature importance is. Depends on task.
- **Neuron specialization**: How specialized individual neurons are. Higher is better.
- **Energy-contribution correlation**: Correlation between energy and importance. Higher suggests good utilization.

## Results Storage

Results are saved to `benchmark_results/` directory:

```
benchmark_results/
├── MNE_benchmark.json
├── StandardMLP_benchmark.json
└── comparison_report.txt
```

## Running Full Benchmark Suite

```bash
cd /z/MNE
python -m pytest tests/benchmark.py -v -s

# Or run directly
python tests/benchmark.py
```

## Benchmarks Included

1. **Adversarial Robustness**
   - FGSM (epsilon=0.0314)
   - PGD-10 (10 steps)
   - PGD-50 (50 steps)
   - AutoAttack (ensemble attack)

2. **OOD Detection**
   - Uses max softmax probability / energy-based OOD scores
   - Compatible with pytorch-ood for accurate metrics

3. **Calibration**
   - Expected Calibration Error (ECE)
   - Reliability diagrams (visualization)

4. **Energy Measurement**
   - GPU energy with Zeus
   - CPU/GPU energy with pyJoules
   - Fallback approximation if neither available

5. **Interpretability**
   - Gradient-based saliency
   - Feature attribution entropy
   - Neuron specialization (MNE-specific)

## Tips for Benchmarking

1. **Data Size**: Use ~5000 samples for quick benchmarks, more for comprehensive results
2. **Device**: Use GPU for faster evaluation if available
3. **Seeds**: Use consistent seeds for reproducibility
4. **External Libraries**: Install all optional libraries for complete evaluation
5. **Energy Measurement**: On CPU/GPU systems, energy tools require specific hardware support

## Citation

If you use this benchmark suite in your research, please cite:

```bibtex
@software{mne_benchmark_2026,
  title = {Comprehensive Benchmark Suite for Metabolic Neural Ecosystem},
  author = {MNE Research Team},
  year = {2026},
  url = {https://github.com/zabwie/mne},
  note = {Evaluates neural networks across generalization, efficiency, stability, continual learning, and interpretability}
}
```

## Questions?

For questions or issues, please open an issue on GitHub: [https://github.com/zabwie/mne/issues](https://github.com/zabwie/mne/issues)

## Troubleshooting

### NumPy Compatibility

The benchmark is fully compatible with both NumPy 1.x and NumPy 2.x. All NumPy compatibility issues have been fixed:

1. **Tensor conversions**: Use `.tolist()` instead of `..numpy()` for PyTorch tensor conversions
2. **Permutation indexing**: Convert NumPy permutations to lists before using them to index tensors
3. **np.trapz vs np.trapezoid**: Auto-detect NumPy version and use the correct function

If you see warnings like:
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.4.2
```

**Don't worry!** These are just warnings from PyTorch. The benchmark will run correctly.

### Benchmark Not Running

If the benchmark crashes or stalls:

1. **Check imports**: Run `python tests/test_benchmark.py` to verify all imports work
2. **Reduce sample size**: Set `num_samples=1000` for faster testing
3. **Disable optional features**: Set `run_continual=False` to skip continual learning tests
4. **Check external libs**: Missing libs like `torchattacks` or `Captum` will skip related tests (not crash)