# cortical_layers.py
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple
from neuron import NeuronType

class CorticalLayer(Enum):
    L1 = 1    # Molecular layer
    L2 = 2    # External granular layer
    L23 = 23  # Mixed layer 2/3
    L4 = 4    # Internal granular layer
    L5A = 51  # Upper layer 5
    L5B = 52  # Lower layer 5
    L6A = 61  # Upper layer 6
    L6B = 62  # Lower layer 6
    WM = 7    # White matter

@dataclass
class LayerProperties:
    thickness: float  # μm
    density: float    # neurons/mm³
    neuron_types: Dict[NeuronType, float]  # type -> proportion
    connectivity_rules: Dict[Tuple[NeuronType, NeuronType], float]  # (pre, post) -> probability

def get_base_layer_properties() -> Dict[CorticalLayer, LayerProperties]:
    """Define detailed properties for each cortical layer."""
    return {
        CorticalLayer.L1: LayerProperties(
            thickness=150.0,
            density=20000,
            neuron_types={
                NeuronType.VIP: 0.45,
                NeuronType.BIPOLAR: 0.35,
                NeuronType.NEUROGLIAFORM: 0.2
            },
            connectivity_rules={
                (NeuronType.VIP, NeuronType.PYRAMIDAL): 0.1,
                (NeuronType.VIP, NeuronType.BIPOLAR): 0.15,
                (NeuronType.NEUROGLIAFORM, NeuronType.PYRAMIDAL): 0.2
            }
        ),
        
        CorticalLayer.L2: LayerProperties(
            thickness=250.0,
            density=90000,
            neuron_types={
                NeuronType.PYRAMIDAL: 0.7,
                NeuronType.BASKET: 0.15,
                NeuronType.MARTINOTTI: 0.1,
                NeuronType.NEUROGLIAFORM: 0.05
            },
            connectivity_rules={
                (NeuronType.PYRAMIDAL, NeuronType.PYRAMIDAL): 0.15,
                (NeuronType.BASKET, NeuronType.PYRAMIDAL): 0.5,
                (NeuronType.MARTINOTTI, NeuronType.PYRAMIDAL): 0.3,
                (NeuronType.PYRAMIDAL, NeuronType.BASKET): 0.2
            }
        ),
        
        CorticalLayer.L23: LayerProperties(
            thickness=500.0,
            density=80000,
            neuron_types={
                NeuronType.PYRAMIDAL: 0.8,
                NeuronType.BASKET: 0.1,
                NeuronType.MARTINOTTI: 0.05,
                NeuronType.CHANDELIER: 0.05
            },
            connectivity_rules={
                (NeuronType.PYRAMIDAL, NeuronType.PYRAMIDAL): 0.2,
                (NeuronType.BASKET, NeuronType.PYRAMIDAL): 0.5,
                (NeuronType.MARTINOTTI, NeuronType.PYRAMIDAL): 0.3,
                (NeuronType.CHANDELIER, NeuronType.PYRAMIDAL): 0.4,
                (NeuronType.PYRAMIDAL, NeuronType.BASKET): 0.25
            }
        ),
        
        CorticalLayer.L4: LayerProperties(
            thickness=300.0,
            density=100000,
            neuron_types={
                NeuronType.SPINY_STELLATE: 0.5,
                NeuronType.PYRAMIDAL: 0.3,
                NeuronType.BASKET: 0.1,
                NeuronType.CHANDELIER: 0.05,
                NeuronType.MARTINOTTI: 0.05
            },
            connectivity_rules={
                (NeuronType.SPINY_STELLATE, NeuronType.PYRAMIDAL): 0.3,
                (NeuronType.SPINY_STELLATE, NeuronType.SPINY_STELLATE): 0.2,
                (NeuronType.PYRAMIDAL, NeuronType.SPINY_STELLATE): 0.25,
                (NeuronType.BASKET, NeuronType.SPINY_STELLATE): 0.4,
                (NeuronType.CHANDELIER, NeuronType.PYRAMIDAL): 0.35
            }
        ),
        
        CorticalLayer.L5A: LayerProperties(
            thickness=400.0,
            density=70000,
            neuron_types={
                NeuronType.PYRAMIDAL: 0.75,
                NeuronType.BASKET: 0.15,
                NeuronType.MARTINOTTI: 0.05,
                NeuronType.BIPOLAR: 0.05
            },
            connectivity_rules={
                (NeuronType.PYRAMIDAL, NeuronType.PYRAMIDAL): 0.2,
                (NeuronType.BASKET, NeuronType.PYRAMIDAL): 0.45,
                (NeuronType.MARTINOTTI, NeuronType.PYRAMIDAL): 0.25,
                (NeuronType.PYRAMIDAL, NeuronType.BASKET): 0.2
            }
        ),
        
        CorticalLayer.L5B: LayerProperties(
            thickness=400.0,
            density=60000,
            neuron_types={
                NeuronType.PYRAMIDAL: 0.8,
                NeuronType.BASKET: 0.1,
                NeuronType.MARTINOTTI: 0.05,
                NeuronType.VIP: 0.05
            },
            connectivity_rules={
                (NeuronType.PYRAMIDAL, NeuronType.PYRAMIDAL): 0.15,
                (NeuronType.BASKET, NeuronType.PYRAMIDAL): 0.4,
                (NeuronType.MARTINOTTI, NeuronType.PYRAMIDAL): 0.3,
                (NeuronType.PYRAMIDAL, NeuronType.MARTINOTTI): 0.2
            }
        ),
        
        CorticalLayer.L6A: LayerProperties(
            thickness=500.0,
            density=65000,
            neuron_types={
                NeuronType.PYRAMIDAL: 0.7,
                NeuronType.BASKET: 0.15,
                NeuronType.MARTINOTTI: 0.1,
                NeuronType.VIP: 0.05
            },
            connectivity_rules={
                (NeuronType.PYRAMIDAL, NeuronType.PYRAMIDAL): 0.1,
                (NeuronType.BASKET, NeuronType.PYRAMIDAL): 0.35,
                (NeuronType.MARTINOTTI, NeuronType.PYRAMIDAL): 0.25,
                (NeuronType.PYRAMIDAL, NeuronType.BASKET): 0.15
            }
        ),
        
        CorticalLayer.L6B: LayerProperties(
            thickness=250.0,
            density=40000,
            neuron_types={
                NeuronType.PYRAMIDAL: 0.6,
                NeuronType.BASKET: 0.2,
                NeuronType.MARTINOTTI: 0.15,
                NeuronType.VIP: 0.05
            },
            connectivity_rules={
                (NeuronType.PYRAMIDAL, NeuronType.PYRAMIDAL): 0.08,
                (NeuronType.BASKET, NeuronType.PYRAMIDAL): 0.3,
                (NeuronType.MARTINOTTI, NeuronType.PYRAMIDAL): 0.2,
                (NeuronType.PYRAMIDAL, NeuronType.MARTINOTTI): 0.15
            }
        )
    }