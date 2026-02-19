"""
Simple demonstration of Metabolic Neural Ecosystem (MNE).

This example shows the basic functionality of MNE without complex training.
"""

import torch
import sys
import os

# Import MNE components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core import MNE, MNEConfig
from src.neuron import MNENeuron
from src.synapse import MNESynapse
from src.energy import MNEEnergyManager
from src.topology import MNETopology


def demonstrate_neuron():
    """Demonstrate neuron functionality."""
    print("=== Neuron Demonstration ===")

    # Create neuron
    neuron = MNENeuron(num_neurons=10)
    state = neuron.get_initial_state(batch_size=4)

    print(f"Initial state:")
    print(f"  Activation shape: {state.activation.shape}")
    print(f"  Resource shape: {state.resource.shape}")
    print(f"  Threshold shape: {state.threshold.shape}")
    print(f"  Is active shape: {state.is_active.shape}")

    # Forward pass
    inputs = torch.randn(4, 10)
    weights = torch.randn(10, 10)
    contribution = torch.rand(4, 10)

    activation, new_state = neuron.forward(inputs, weights, state, contribution)

    print(f"\nAfter forward pass:")
    print(f"  Activation range: [{activation.min():.3f}, {activation.max():.3f}]")
    print(
        f"  Resource range: [{new_state.resource.min():.3f}, {new_state.resource.max():.3f}]"
    )
    print(f"  Active neurons: {new_state.is_active.sum().item()}/40")

    return neuron, new_state


def demonstrate_synapse():
    """Demonstrate synapse functionality."""
    print("\n=== Synapse Demonstration ===")

    # Create synapse
    synapse = MNESynapse(num_neurons=10)
    state = synapse.get_initial_state()

    print(f"Initial state:")
    print(f"  Weights shape: {state.weights.shape}")
    print(f"  Connection sparsity: {1 - state.is_connected.float().mean():.2%}")

    # Update
    presynaptic = torch.randn(4, 10)
    postsynaptic = torch.randn(4, 10)
    contribution = torch.rand(4, 10)

    new_state = synapse.update(presynaptic, postsynaptic, contribution, state)

    print(f"\nAfter update:")
    print(f"  Weight change: {(new_state.weights - state.weights).abs().mean():.4f}")
    print(
        f"  Energy cost range: [{new_state.energy_cost.min():.4f}, {new_state.energy_cost.max():.4f}]"
    )

    return synapse, new_state


def demonstrate_energy():
    """Demonstrate energy management."""
    print("\n=== Energy Management Demonstration ===")

    # Create energy manager
    energy_manager = MNEEnergyManager(
        initial_energy=100.0,
        energy_influx=10.0,
        min_energy=0.0,
        max_energy=200.0,
    )
    state = energy_manager.get_initial_state()

    print(f"Initial state:")
    print(f"  Total energy: {state.total_energy:.2f}")
    print(f"  Efficiency: {state.efficiency:.4f}")

    # Update energy
    neuron_consumption = torch.randn(4, 10).abs()  # Must be non-negative
    contribution = torch.rand(4, 10)

    new_state = energy_manager.update(neuron_consumption, contribution, state)

    print(f"\nAfter update:")
    print(f"  Total energy: {new_state.total_energy:.2f}")
    print(f"  Efficiency: {new_state.efficiency:.4f}")
    print(f"  Is constrained: {new_state.is_constrained}")

    return energy_manager, new_state


def demonstrate_topology():
    """Demonstrate topology management."""
    print("\n=== Topology Demonstration ===")

    # Create topology
    topology = MNETopology(
        max_neurons=20,
        min_neurons=5,
        resource_high=8.0,
        resource_low=2.0,
    )

    # Create neuron state with varying resources
    neuron_state = type("NeuronState", (), {})()
    neuron_state.resource = torch.tensor(
        [[1.0, 9.0, 3.0, 7.0, 0.5, 8.5, 2.5, 6.0, 1.5, 10.0]]
    )
    neuron_state.is_active = torch.ones(1, 10, dtype=torch.bool)
    neuron_state.activation = torch.randn(1, 10)
    neuron_state.threshold = torch.zeros(1, 10)
    neuron_state.age = torch.zeros(1, 10)

    # Create synapse state
    synapse_state = type("SynapseState", (), {})()
    synapse_state.weights = torch.randn(10, 10)
    synapse_state.is_connected = torch.rand(10, 10) > 0.5
    synapse_state.energy_cost = torch.rand(10, 10)
    synapse_state.age = torch.zeros(10, 10)

    # Create topology state
    topology_state = type("TopologyState", (), {})()
    topology_state.num_neurons = 10
    topology_state.num_active = 10
    topology_state.neuron_indices = torch.arange(10)
    topology_state.neurogenesis_count = 0
    topology_state.apoptosis_count = 0
    topology_state.homeostatic_state = {}

    print(f"Initial state:")
    print(f"  Neurons: {topology_state.num_neurons}")
    print(f"  Active: {topology_state.num_active}")
    print(f"  Resources: {neuron_state.resource[0].tolist()}")

    # Check for neurogenesis/apoptosis
    split_mask, death_mask = topology.check_neurogenesis_apoptosis(neuron_state)

    print(f"\nNeurogenesis candidates: {split_mask.sum().item()} neurons")
    print(f"Apoptosis candidates: {death_mask.sum().item()} neurons")

    return topology


def demonstrate_full_mne():
    """Demonstrate full MNE system."""
    print("\n=== Full MNE System Demonstration ===")

    # Configure MNE
    config = MNEConfig(
        num_neurons=20,
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
        max_neurons=30,
        min_neurons=10,
        resource_high=8.0,
        resource_low=2.0,
        # System parameters
        device="cpu",
    )

    # Create MNE
    mne = MNE(config)
    print(f"MNE created with {config.num_neurons} neurons")

    # Get initial state
    state = mne.get_initial_state(batch_size=4)
    print(f"Initial state time step: {state.time_step}")

    # Forward pass
    inputs = torch.randn(4, config.num_neurons)
    outputs, new_state = mne(inputs, state)

    print(f"\nAfter forward pass:")
    print(f"  Outputs shape: {outputs.shape}")
    print(f"  New time step: {new_state.time_step}")
    print(f"  Energy: {new_state.energy_state.total_energy:.2f}")
    print(f"  Efficiency: {new_state.energy_state.efficiency:.4f}")

    # Multiple steps
    print(f"\nRunning 5 time steps...")
    current_state = new_state
    for i in range(5):
        inputs = torch.randn(4, config.num_neurons)
        outputs, current_state = mne(inputs, current_state)

        if i % 2 == 0:
            metrics = mne.get_metrics(current_state)
            print(
                f"  Step {i + 1}: Neurons={metrics['num_neurons']}, "
                f"Active={metrics['num_active']}, "
                f"Energy={metrics['total_energy']:.2f}"
            )

    return mne, current_state


def main():
    """Main demonstration function."""
    print("Metabolic Neural Ecosystem (MNE) Demonstration")
    print("=" * 50)

    # Demonstrate individual components
    neuron, neuron_state = demonstrate_neuron()
    synapse, synapse_state = demonstrate_synapse()
    energy_manager, energy_state = demonstrate_energy()
    topology = demonstrate_topology()

    # Demonstrate full system
    mne, final_state = demonstrate_full_mne()

    print("\n" + "=" * 50)
    print("Demonstration Complete!")
    print("\nKey Features Demonstrated:")
    print("1. Neurons with metabolic state tracking")
    print("2. Synapses with energy-aware plasticity")
    print("3. Global energy budget management")
    print("4. Dynamic topology (neurogenesis/apoptosis)")
    print("5. Integrated MNE system")

    # Show final metrics
    final_metrics = mne.get_metrics(final_state)
    print(f"\nFinal System State:")
    print(f"  Time steps: {final_state.time_step}")
    print(f"  Total neurons: {final_metrics['num_neurons']}")
    print(f"  Active neurons: {final_metrics['num_active']}")
    print(f"  Total energy consumed: {final_metrics['total_energy']:.2f}")
    print(f"  Average efficiency: {final_metrics['avg_efficiency']:.4f}")
    print(f"  Neurogenesis events: {final_metrics['neurogenesis_count']}")
    print(f"  Apoptosis events: {final_metrics['apoptosis_count']}")


if __name__ == "__main__":
    main()
