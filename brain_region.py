# brain_region.py
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class BrainRegion:
    """Represents a brain region with spatial properties and neuron indices."""
    name: str
    num_neurons: int
    center: Tuple[float, float, float]
    radius: float
    layers: int = 6  # Number of cortical layers
    start_idx: int = 0
    end_idx: int = 0
    layer_indices: Dict[int, Tuple[int, int]] = None  # Start and end indices for each layer
