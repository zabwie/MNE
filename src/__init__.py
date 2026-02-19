"""
Metabolic Neural Ecosystem (MNE) - A biologically inspired neural network architecture
with metabolic constraints, energy-aware plasticity, and structural plasticity.

This module implements the core MNE architecture based on research findings from
metaboplasticity, multi-scale homeostasis, and energy-aware Hebbian learning.

Key Components:
- Neuron: Individual neurons with metabolic state tracking
- Synapse: Energy-aware synaptic connections with plasticity
- Energy: Global energy budget management
- Topology: Structural plasticity (neurogenesis/apoptosis) and homeostasis
- Core: Main MNE orchestrator

Mathematical Foundation:
The MNE architecture implements the following core equations:

1. Activation update:
   a_i(t+1) = f(∑_j w_ij(t) a_j(t) + I_i(t))

2. Contribution (gradient-based):
   contrib_i(t) = |∂L/∂a_i|

3. Consumption:
   consume_i(t) = κ a_i(t)² + ∑_j γ |w_ij(t)| a_j(t)

4. Resource update:
   r_i(t+1) = r_i(t) + α·contrib_i(t) - β·consume_i(t) - δ r_i(t)

5. Weight update (energy-aware Hebbian):
   w_ij(t+1) = w_ij(t) + η·contrib_i(t)·a_i(t)a_j(t) - μγ|w_ij(t)|a_j(t)w_ij(t)

6. Homeostasis:
   θ_i(t+1) = θ_i(t) + ρ(a_i(t) - a_target)

7. Neurogenesis:
   If r_i(t) > R_high, split neuron

8. Apoptosis:
   If r_i(t) < R_low, kill neuron

9. Global energy:
   E_total(t+1) = E_total(t) + E_influx - ∑_i consume_i(t)
"""

__version__ = "0.1.0"
__author__ = "MNE Research Team"

from .neuron import MNENeuron, NeuronState
from .synapse import MNESynapse, SynapseState
from .energy import MNEEnergyManager, EnergyState
from .topology import MNETopology, HomeostaticRegulator
from .core import MNE, MNEConfig

__all__ = [
    # Core classes
    "MNE",
    "MNEConfig",
    # Neuron components
    "MNENeuron",
    "NeuronState",
    # Synapse components
    "MNESynapse",
    "SynapseState",
    # Energy components
    "MNEEnergyManager",
    "EnergyState",
    # Topology components
    "MNETopology",
    "HomeostaticRegulator",
]
