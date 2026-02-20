"""
MNE - High-Performance Metabolic Neural Ecosystem.

This version is engineered for MAXIMUM accuracy:
1. 4x larger network (256 neurons, 4 layers)
2. More aggressive regularization
3. Better initialization
4. Extended training with advanced LR scheduling
5. More robust optimization
6. Enhanced data augmentation capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from .neuron import MNENeuron, NeuronState
from .synapse import MNESynapse, SynapseState
from .energy import MNEEnergyManager, EnergyState
from .topology import MNETopology, TopologyState


@dataclass
class MNELayerState:
    """State for a single MNE recurrent layer."""

    neuron_state: NeuronState
    synapse_state: SynapseState


class MNELayer(nn.Module):
    """
    High-performance MNE layer with enhanced features.
    """

    def __init__(
        self,
        num_neurons: int,
        layer_idx: int,
        activation_fn: str = "leaky_relu",
        dropout_rate: float = 0.2,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.layer_idx = layer_idx

        # MNE components with optimized parameters
        self.neuron = MNENeuron(
            num_neurons=num_neurons,
            activation_fn=activation_fn,
            device=device,
        )
        self.synapse = MNESynapse(
            num_neurons=num_neurons,
            device=device,
        )

        # Layer normalization with more aggressive settings
        self.layer_norm = nn.LayerNorm(num_neurons, device=device)

        # Pre-norm for better gradient flow
        self.pre_norm = nn.LayerNorm(num_neurons, device=device)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Gating mechanism for learnable residual weighting
        self.gate = nn.Linear(num_neurons, num_neurons, device=device)
        self.gate_activation = nn.Sigmoid()

        self.to(device)

    def forward(
        self,
        inputs: torch.Tensor,
        layer_state: MNELayerState,
        contribution: Optional[torch.Tensor] = None,
        apply_plasticity: bool = True,
    ) -> Tuple[torch.Tensor, MNELayerState]:
        """Forward pass with pre-norm, residual, and gating."""
        batch_size = inputs.shape[0]

        # Pre-norm
        x_norm = self.pre_norm(inputs)

        # MNE forward
        weights = self.synapse.get_weights(layer_state.synapse_state)
        activation, neuron_state = self.neuron(
            x_norm, weights, layer_state.neuron_state, contribution
        )

        # Update synapses
        if apply_plasticity and contribution is not None:
            synapse_state = self.synapse.update(
                activation, activation, contribution, layer_state.synapse_state
            )
        else:
            synapse_state = layer_state.synapse_state

        # Gated residual connection
        gate_values = self.gate_activation(self.gate(inputs))
        residual = gate_values * activation + (1 - gate_values) * inputs

        # Layer norm
        output = self.layer_norm(residual)

        # Dropout
        output = self.dropout(output)

        new_state = MNELayerState(
            neuron_state=neuron_state,
            synapse_state=synapse_state,
        )

        return output, new_state

    def get_initial_state(self, batch_size: int) -> MNELayerState:
        """Initialize layer state."""
        return MNELayerState(
            neuron_state=self.neuron.get_initial_state(batch_size),
            synapse_state=self.synapse.get_initial_state(),
        )


@dataclass
class MNEConfig:
    """
    Configuration for MNE.
    """

    # Architecture - MAKE IT BIG!
    num_neurons: int = 256  # Large capacity
    input_dim: int = 784
    output_dim: int = 10
    num_layers: int = 4  # Deep network
    activation_fn: str = "leaky_relu"

    # Regularization - Strong
    dropout_rate: float = 0.25
    weight_decay: float = 0.02

    # Training - Thorough
    gradient_lr: float = 0.0007
    total_epochs: int = 30  # Much more training
    warmup_epochs: int = 5
    label_smoothing: float = 0.05

    # Neuron parameters - Optimized for capacity
    kappa: float = 0.015
    gamma: float = 0.008
    alpha: float = 3.0  # Strong resource gain
    beta: float = 0.1
    delta: float = 0.001
    rho: float = 0.12
    target_activation: float = 0.1
    initial_resource: float = 5.0  # Very rich initial resources
    initial_threshold: float = 0.0

    # Synapse parameters - Strong learning
    eta: float = 0.25  # Very strong Hebbian learning
    mu: float = 0.001  # Minimal metabolic penalty
    weight_init_std: float = 0.015
    weight_clip_min: float = -3.0
    weight_clip_max: float = 3.0
    formation_threshold: float = 0.15
    elimination_threshold: float = 0.015
    sparsity: float = 0.4  # More connected

    # Energy parameters - Permissive
    initial_energy: float = 3000.0
    energy_influx: float = 500.0
    min_energy: float = 300.0
    max_energy: float = 6000.0

    # Topology
    max_neurons: int = 1000
    min_neurons: int = 10
    resource_high: float = 20.0
    resource_low: float = 0.001

    # Gradients
    use_gradient_descent: bool = True
    metabolic_lr_modulation: bool = True
    grad_clip: float = 0.5  # Stronger clipping

    # System parameters
    device: str = "cpu"


@dataclass
class MNEState:
    """Complete state of BEAST MNE."""

    layer_states: List[MNELayerState]
    energy_state: EnergyState
    topology_state: TopologyState
    time_step: int = 0


class MNE(nn.Module):
    """
    Metabolic Neural Ecosystem.

    Designed to absolutely DOMINATE with 70%+ accuracy:
    - 256 neurons per layer (large capacity)
    - 4 deep layers
    - Gated residual connections
    - Pre-norm architecture
    - Strong regularization
    - Extended training (30 epochs)
    - Advanced learning rate scheduling
    """

    def __init__(self, config: MNEConfig):
        super().__init__()
        self.config = config

        # Input projection with strong initialization
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.num_neurons),
            nn.LayerNorm(config.num_neurons),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
        )

        # Kaiming He initialization for ReLU-family activations
        nn.init.kaiming_normal_(
            self.input_proj[0].weight, mode="fan_out", nonlinearity="relu"
        )

        # 4 MNE layers for deep learning capacity
        self.layers = nn.ModuleList(
            [
                MNELayer(
                    num_neurons=config.num_neurons,
                    layer_idx=i,
                    activation_fn=config.activation_fn,
                    dropout_rate=config.dropout_rate,
                    device=config.device,
                )
                for i in range(config.num_layers)
            ]
        )

        # Inter-layer skip connections (optional, for very deep networks)
        self.layer_fusion = nn.ModuleList(
            [
                nn.Linear(config.num_neurons, config.num_neurons)
                for _ in range(config.num_layers - 1)
            ]
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.num_neurons, config.num_neurons),
            nn.LayerNorm(config.num_neurons),
            nn.GELU(),  # Better than ReLU for deeper networks
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.num_neurons, config.num_neurons // 2),
            nn.LayerNorm(config.num_neurons // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.num_neurons // 2, config.output_dim),
        )

        # Initialize output projection (only Linear layers)
        nn.init.kaiming_normal_(
            self.output_proj[0].weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.kaiming_normal_(
            self.output_proj[4].weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.xavier_uniform_(self.output_proj[8].weight, gain=0.1)
        nn.init.zeros_(self.output_proj[8].bias)

        # Move everything to device
        self.layer_fusion.to(config.device)
        self.output_proj.to(config.device)

        # Energy manager
        self.energy_manager = MNEEnergyManager(
            initial_energy=config.initial_energy,
            energy_influx=config.energy_influx,
            min_energy=config.min_energy,
            max_energy=config.max_energy,
            device=config.device,
        )

        # Topology
        self.topology = MNETopology(
            max_neurons=config.max_neurons,
            min_neurons=config.min_neurons,
            resource_high=config.resource_high,
            resource_low=config.resource_low,
            device=config.device,
        )

        self.to(config.device)

        # For learning rate scheduling
        self.current_epoch = 0

    def get_initial_state(self, batch_size: int) -> MNEState:
        """Initialize all layer states."""
        layer_states = [layer.get_initial_state(batch_size) for layer in self.layers]

        return MNEState(
            layer_states=layer_states,
            energy_state=self.energy_manager.get_initial_state(),
            topology_state=self.topology.get_initial_state(self.config.num_neurons),
            time_step=0,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        state: MNEState,
        contribution: Optional[torch.Tensor] = None,
        apply_plasticity: bool = True,
    ) -> Tuple[torch.Tensor, MNEState]:
        """Forward pass through all layers."""
        batch_size = inputs.shape[0]

        # Project inputs
        x = self.input_proj(inputs)

        # Multi-layer MNE pass with skip connections
        layer_states = []
        layer_outputs = []

        for i, layer in enumerate(self.layers):
            x, layer_state = layer(
                x, state.layer_states[i], contribution, apply_plasticity
            )
            layer_states.append(layer_state)
            layer_outputs.append(x)

            # Add inter-layer fusion (skip connection to previous layer)
            if i > 0:
                x = x + self.layer_fusion[i - 1](layer_outputs[i - 1])

        # Output projection
        logits = self.output_proj(x)

        # Update energy
        if contribution is not None and len(layer_states) > 0:
            consumption = self.layers[0].neuron.get_consumption_levels(
                layer_states[0].neuron_state
            )
            energy_state = self.energy_manager.update(
                consumption,
                contribution,
                state.energy_state,
                torch.zeros_like(consumption),
            )
        else:
            energy_state = state.energy_state

        new_state = MNEState(
            layer_states=layer_states,
            energy_state=energy_state,
            topology_state=state.topology_state,
            time_step=state.time_step + 1,
        )

        return logits, new_state

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        state: MNEState,
        optimizer: torch.optim.Optimizer,
        **kwargs,
    ) -> Tuple[torch.Tensor, MNEState, Dict[str, Any]]:
        """Single training step."""

        # Forward pass without plasticity
        self.train()
        outputs, new_state = self.forward(inputs, state, apply_plasticity=False)

        # Compute loss with label smoothing
        loss_fn = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        loss = loss_fn(outputs, targets)

        # Compute contribution as activation magnitude
        with torch.no_grad():
            contrib_list = []
            for layer_state in new_state.layer_states:
                norm_activation = torch.abs(layer_state.neuron_state.activation)
                contrib = norm_activation  # Use as contribution signal
                contrib_list.append(contrib)

            # Average across layers
            if contrib_list:
                contribution = torch.stack(contrib_list, dim=0).mean(dim=0)
            else:
                contribution = None

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Strong gradient clipping for stability
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), max_norm=self.config.grad_clip
            )

        # Adaptive learning rate modulation
        if self.config.metabolic_lr_modulation and len(new_state.layer_states) > 0:
            avg_resource = new_state.layer_states[0].neuron_state.resource.mean().item()
            target_resource = self.config.initial_resource
            lr_modifier = min(max(avg_resource / target_resource, 0.6), 1.4)
        else:
            lr_modifier = 1.0

        # Apply LR modification
        for param in self.parameters():
            if param.grad is not None:
                param.grad.mul_(lr_modifier)

        optimizer.step()

        # Update with metabolic plasticity (no gradients)
        with torch.no_grad():
            self.eval()
            outputs_plastic, final_state = self.forward(
                inputs, new_state, contribution=contribution, apply_plasticity=True
            )
            self.train()

        # Metrics
        with torch.no_grad():
            predictions = outputs.argmax(dim=1)
            accuracy = (predictions == targets).float().mean().item()

        metrics = self.get_metrics(final_state)
        metrics["accuracy"] = accuracy
        metrics["loss"] = loss.item()
        metrics["lr_modifier"] = lr_modifier

        return loss, final_state, metrics

    def get_metrics(self, state: MNEState) -> Dict[str, Any]:
        """Get current metrics."""
        metrics = {
            "time_step": state.time_step,
            "total_energy": self.energy_manager.get_total_energy(
                state.energy_state
            ).item(),
            "efficiency": self.energy_manager.get_efficiency(state.energy_state).item(),
        }

        if len(state.layer_states) > 0:
            metrics["avg_resource"] = (
                state.layer_states[0].neuron_state.resource.mean().item()
            )

        return metrics

    def get_onecycle_lr(self, optimizer, epoch: int):
        """Get one-cycle learning rate."""
        if epoch < self.config.warmup_epochs:
            # Warmup
            lr = self.config.gradient_lr * (epoch + 1) / self.config.warmup_epochs
        elif (
            epoch
            < self.config.warmup_epochs
            + (self.config.total_epochs - self.config.warmup_epochs) // 2
        ):
            # Peak
            lr = self.config.gradient_lr
        else:
            # Decay
            decay_progress = (
                epoch
                - self.config.warmup_epochs
                - (self.config.total_epochs - self.config.warmup_epochs) // 2
            )
            max_decay = (
                self.config.total_epochs
                - self.config.warmup_epochs
                - (self.config.total_epochs - self.config.warmup_epochs) // 2
            )
            lr = self.config.gradient_lr * (1 - decay_progress / max_decay) * 0.1

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        return lr
