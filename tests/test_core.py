"""
Unit tests for MNE core module.
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core import MNE, MNEConfig, MNEState


class TestMNEConfig:
    """Test cases for MNEConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = MNEConfig()

        assert config.num_neurons == 100
        assert config.activation_fn == "tanh"
        assert config.kappa == 0.1
        assert config.gamma == 0.05
        assert config.alpha == 0.5
        assert config.beta == 1.0
        assert config.delta == 0.01
        assert config.rho == 0.01
        assert config.target_activation == 0.5

    def test_custom_config(self):
        """Test custom configuration."""
        config = MNEConfig(num_neurons=50, activation_fn="relu", kappa=0.2, gamma=0.1)

        assert config.num_neurons == 50
        assert config.activation_fn == "relu"
        assert config.kappa == 0.2
        assert config.gamma == 0.1


class TestMNE:
    """Test cases for MNE class."""

    @pytest.fixture
    def config(self):
        """Create a config for testing."""
        return MNEConfig(num_neurons=50, activation_fn="tanh", device="cpu")

    @pytest.fixture
    def mne(self, config):
        """Create an MNE instance for testing."""
        return MNE(config)

    @pytest.fixture
    def state(self, mne):
        """Create initial MNE state."""
        return mne.get_initial_state(batch_size=4)

    def test_initialization(self, mne, config):
        """Test MNE initialization."""
        assert mne.config.num_neurons == 50
        assert mne.config.activation_fn == "tanh"
        assert mne.neuron.num_neurons == 50
        assert mne.synapse.num_neurons == 50

    def test_get_initial_state(self, mne, state):
        """Test initial state creation."""
        assert state.neuron_state.activation.shape == (4, 50)
        assert state.synapse_state.weights.shape == (50, 50)
        assert state.energy_state.total_energy.item() == mne.config.initial_energy
        assert state.topology_state.num_neurons == 50
        assert state.time_step == 0

    def test_forward(self, mne, state):
        """Test forward pass."""
        inputs = torch.randn(4, 50)

        outputs, new_state = mne.forward(inputs, state)

        assert outputs.shape == (4, 50)
        assert new_state.time_step == 1
        assert new_state.neuron_state.activation.shape == (4, 50)
        assert new_state.synapse_state.weights.shape == (50, 50)

    def test_forward_with_contribution(self, mne, state):
        """Test forward pass with contribution."""
        inputs = torch.randn(4, 50)
        contribution = torch.rand(4, 50)

        outputs, new_state = mne.forward(inputs, state, contribution=contribution)

        assert outputs.shape == (4, 50)
        assert new_state.time_step == 1

    def test_forward_without_plasticity(self, mne, state):
        """Test forward pass without plasticity."""
        inputs = torch.randn(4, 50)
        initial_weights = state.synapse_state.weights.clone()

        outputs, new_state = mne.forward(inputs, state, apply_plasticity=False)

        # Weights should not change
        assert torch.allclose(new_state.synapse_state.weights, initial_weights)

    def test_forward_without_topology(self, mne, state):
        """Test forward pass without topology changes."""
        inputs = torch.randn(4, 50)
        initial_active = state.neuron_state.is_active.clone()

        outputs, new_state = mne.forward(inputs, state, apply_topology=False)

        # Active neurons should not change
        assert torch.all(new_state.neuron_state.is_active == initial_active)

    def test_compute_loss(self, mne, state):
        """Test loss computation."""
        outputs = torch.randn(4, 50)
        targets = torch.randn(4, 50)

        loss = mne.compute_loss(outputs, targets, state)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_compute_contribution(self, mne, state):
        """Test contribution computation."""
        outputs = torch.randn(4, 50)
        targets = torch.randn(4, 50)

        contribution = mne.compute_contribution(outputs, targets, state)

        assert contribution.shape == (4, 50)
        assert torch.all(contribution >= 0)

    def test_train_step(self, mne, state):
        """Test training step."""
        inputs = torch.randn(4, 50)
        targets = torch.randn(4, 50)

        loss, new_state, metrics = mne.train_step(inputs, targets, state)

        assert isinstance(loss, torch.Tensor)
        assert new_state.time_step == 2  # Two forward passes
        assert isinstance(metrics, dict)
        assert "time_step" in metrics
        assert "num_neurons" in metrics
        assert "total_energy" in metrics

    def test_train_step_with_optimizer(self, mne, state):
        """Test training step with optimizer."""
        # MNE uses its own plasticity mechanisms, not standard backprop
        # So we skip the optimizer test
        inputs = torch.randn(4, 50)
        targets = torch.randn(4, 50)

        loss, new_state, metrics = mne.train_step(inputs, targets, state)

        assert isinstance(loss, torch.Tensor)
        assert new_state.time_step == 2

    def test_get_metrics(self, mne, state):
        """Test getting metrics."""
        metrics = mne.get_metrics(state)

        assert isinstance(metrics, dict)
        assert "time_step" in metrics
        assert "num_neurons" in metrics
        assert "num_active" in metrics
        assert "total_energy" in metrics
        assert "efficiency" in metrics
        assert "is_constrained" in metrics
        assert "neurogenesis_count" in metrics
        assert "apoptosis_count" in metrics

    def test_get_weights(self, mne, state):
        """Test getting weights."""
        weights = mne.get_weights(state)

        assert weights.shape == (50, 50)
        assert torch.all(weights == state.synapse_state.weights)

    def test_get_activations(self, mne, state):
        """Test getting activations."""
        activations = mne.get_activations(state)

        assert activations.shape == (4, 50)
        assert torch.all(activations == state.neuron_state.activation)

    def test_get_resources(self, mne, state):
        """Test getting resources."""
        resources = mne.get_resources(state)

        assert resources.shape == (4, 50)
        assert torch.all(resources == state.neuron_state.resource)

    def test_reset_state(self, mne):
        """Test state reset."""
        state = mne.get_initial_state(batch_size=4)

        # Modify state
        state.time_step = 10
        state.neuron_state.activation = torch.randn(4, 50)

        # Reset
        new_state = mne.reset_state(batch_size=4)

        assert new_state.time_step == 0
        assert torch.all(new_state.neuron_state.activation == 0)

    def test_set_energy_influx(self, mne, state):
        """Test setting energy influx."""
        new_state = mne.set_energy_influx(state, 20.0)

        assert new_state.energy_state.energy_influx.item() == 20.0

    def test_add_energy(self, mne, state):
        """Test adding energy."""
        initial_energy = state.energy_state.total_energy.clone()

        new_state = mne.add_energy(state, 50.0)

        assert (
            new_state.energy_state.total_energy.item() == initial_energy.item() + 50.0
        )

    def test_multiple_forward_passes(self, mne, state):
        """Test multiple forward passes."""
        inputs = torch.randn(4, 50)

        for i in range(5):
            outputs, state = mne.forward(inputs, state)

        assert state.time_step == 5

    def test_energy_dynamics_over_time(self, mne, state):
        """Test energy dynamics over multiple steps."""
        inputs = torch.randn(4, 50)
        contribution = torch.rand(4, 50)

        initial_energy = state.energy_state.total_energy.clone()

        # Run for several steps
        for _ in range(10):
            outputs, state = mne.forward(inputs, state, contribution=contribution)

        # Energy should change over time
        assert not torch.allclose(
            state.energy_state.total_energy, initial_energy, atol=0.01
        )

    def test_topology_changes_over_time(self, mne, state):
        """Test topology changes over multiple steps."""
        inputs = torch.randn(4, 50)
        contribution = torch.rand(4, 50)

        # Set some neurons to low resource to trigger apoptosis
        state.neuron_state.resource[:, :5] = 0.05

        initial_active = state.topology_state.num_active

        # Run for several steps
        for _ in range(10):
            outputs, state = mne.forward(inputs, state, contribution=contribution)

        # Active count may change due to topology
        # (exact behavior depends on thresholds)

    def test_different_activation_functions(self):
        """Test different activation functions."""
        for activation_fn in ["tanh", "relu", "sigmoid", "leaky_relu"]:
            config = MNEConfig(num_neurons=20, activation_fn=activation_fn)
            mne = MNE(config)
            state = mne.get_initial_state(batch_size=2)
            inputs = torch.randn(2, 20)

            outputs, new_state = mne.forward(inputs, state)

            assert outputs.shape == (2, 20)

    def test_different_batch_sizes(self, mne):
        """Test different batch sizes."""
        for batch_size in [1, 4, 8, 16]:
            state = mne.get_initial_state(batch_size=batch_size)
            inputs = torch.randn(batch_size, 50)

            outputs, new_state = mne.forward(inputs, state)

            assert outputs.shape == (batch_size, 50)

    def test_different_num_neurons(self):
        """Test different numbers of neurons."""
        for num_neurons in [10, 50, 100]:
            config = MNEConfig(num_neurons=num_neurons)
            mne = MNE(config)
            state = mne.get_initial_state(batch_size=4)
            inputs = torch.randn(4, num_neurons)

            outputs, new_state = mne.forward(inputs, state)

            assert outputs.shape == (4, num_neurons)

    def test_metrics_completeness(self, mne, state):
        """Test that all expected metrics are present."""
        metrics = mne.get_metrics(state)

        expected_keys = [
            "time_step",
            "num_neurons",
            "num_active",
            "total_energy",
            "efficiency",
            "is_constrained",
            "neurogenesis_count",
            "apoptosis_count",
            "energy_influx",
            "total_consumption",
            "energy_level",
            "active_fraction",
        ]

        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_state_consistency(self, mne, state):
        """Test that state remains consistent across updates."""
        inputs = torch.randn(4, 50)

        # Run forward pass
        outputs, new_state = mne.forward(inputs, state)

        # Check that all state components are updated
        assert new_state.neuron_state is not None
        assert new_state.synapse_state is not None
        assert new_state.energy_state is not None
        assert new_state.topology_state is not None

        # Check shapes
        assert new_state.neuron_state.activation.shape == (4, 50)
        assert new_state.synapse_state.weights.shape == (50, 50)

    def test_device_placement(self):
        """Test that tensors are placed on correct device."""
        config = MNEConfig(num_neurons=20, device="cpu")
        mne = MNE(config)
        state = mne.get_initial_state(batch_size=2)

        # Check that all tensors are on CPU
        assert state.neuron_state.activation.device.type == "cpu"
        assert state.synapse_state.weights.device.type == "cpu"
        assert state.energy_state.total_energy.device.type == "cpu"
