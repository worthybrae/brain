# digital_brain.py
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import os

from cortical_layers import CorticalLayer, LayerProperties, get_base_layer_properties
from neuron import AdvancedNeuron
from connection import AdvancedConnection
from brain_region import BrainRegion

@dataclass
class RegionLayerConfig:
    """Configuration for a cortical layer within a brain region."""
    layer: CorticalLayer
    properties: LayerProperties
    start_idx: int = 0
    end_idx: int = 0
    neurons: List[AdvancedNeuron] = None
    positions: np.ndarray = None

def _initialize_layer_neurons_chunk(args):
    """Worker function to initialize neurons in parallel."""
    start_idx, end_idx, layer_properties = args
    neurons = []
    for idx in range(start_idx, end_idx):
        neuron_type = _select_neuron_type_worker(layer_properties.neuron_types)
        neurons.append((idx, AdvancedNeuron(idx, neuron_type)))
    return neurons

def _select_neuron_type_worker(type_proportions):
    """Worker function for neuron type selection."""
    rand_val = np.random.random()
    cumulative_prob = 0.0
    for n_type, proportion in type_proportions.items():
        cumulative_prob += proportion
        if rand_val <= cumulative_prob:
            return n_type
    return list(type_proportions.keys())[0]

def _process_connections_chunk(args):
    """Worker function to process connections in parallel."""
    source_neurons, target_neurons, base_prob, distances = args
    connections = []
    
    # Vectorized distance-based probability calculation
    connection_probs = base_prob * np.exp(-distances / 50.0)
    random_vals = np.random.random(connection_probs.shape)
    valid_connections = np.where(random_vals < connection_probs)
    
    for i, j in zip(*valid_connections):
        source_idx = source_neurons[i].idx
        target_idx = target_neurons[j].idx
        distance = distances[i, j]
        
        # Create connection with distance-dependent properties
        weight = 0.5 + 0.1 * np.exp(-distance / 100.0)
        delay = 1.0 + distance * 0.1  # Distance-dependent delay
        # Return just the connection object, not a tuple
        connections.append(AdvancedConnection(source_idx, target_idx, weight, delay))
    
    return connections

def _position_neurons_chunk(args):
    """Worker function to position neurons in parallel."""
    start_idx, chunk_size, center, radius = args
    theta = np.random.uniform(0, 2 * np.pi, chunk_size)
    phi = np.arccos(np.random.uniform(-1, 1, chunk_size))
    r = radius * np.cbrt(np.random.uniform(0, 1, chunk_size))

    x = r * np.sin(phi) * np.cos(theta) + center[0]
    y = r * np.sin(phi) * np.sin(theta) + center[1]
    z = r * np.cos(phi) + center[2]

    return start_idx, np.column_stack((x, y, z))

class EnhancedBrainRegion(BrainRegion):
    """Enhanced brain region with cortical layer support."""
    def __init__(self, name: str, num_neurons: int, center: Tuple[float, float, float], 
                 radius: float):
        super().__init__(name, num_neurons, center, radius)
        self.layer_configs: Dict[CorticalLayer, RegionLayerConfig] = {}
        self._initialize_layers()
    
    def _initialize_layers(self):
        """Initialize cortical layers with proper proportions."""
        base_properties = get_base_layer_properties()
        total_thickness = sum(props.thickness for props in base_properties.values())
        
        current_idx = self.start_idx
        for layer, props in base_properties.items():
            layer_neurons = int(self.num_neurons * (props.thickness / total_thickness))
            
            config = RegionLayerConfig(
                layer=layer,
                properties=props,
                start_idx=current_idx,
                end_idx=current_idx + layer_neurons
            )
            
            self.layer_configs[layer] = config
            current_idx += layer_neurons

class ModularDigitalBrain:
    """Enhanced digital brain with modular architecture and cortical layer support."""
    
    def __init__(self, scale_factor: float = 0.01):
        self.scale_factor = scale_factor
        self.regions: Dict[str, EnhancedBrainRegion] = self._initialize_regions()
        self.neurons: List[AdvancedNeuron] = []
        self.connections: Dict[Tuple[int, int], AdvancedConnection] = {}
        
        print("Initializing digital brain...")
        with tqdm(total=4, desc="Setup progress") as pbar:
            self._initialize_neuron_positions()
            pbar.update(1)
            
            self._initialize_neurons()
            pbar.update(1)
            
            self._initialize_connections()
            pbar.update(1)
            
            self._validate_brain()
            pbar.update(1)
    
    def _initialize_regions(self) -> Dict[str, EnhancedBrainRegion]:
        """Initialize brain regions with cortical layers."""
        print("Initializing brain regions...")
        regions = {
            'v1': EnhancedBrainRegion('v1', int(2_000_000 * self.scale_factor), (0, -80, 0), 20),
            'v2': EnhancedBrainRegion('v2', int(1_000_000 * self.scale_factor), (0, -70, 0), 15),
            'v4': EnhancedBrainRegion('v4', int(1_000_000 * self.scale_factor), (30, -60, -5), 12),
            'it': EnhancedBrainRegion('it', int(2_000_000 * self.scale_factor), (40, -50, -10), 15),
            'pfc': EnhancedBrainRegion('pfc', int(4_000_000 * self.scale_factor), (30, 50, 0), 25)
        }
        return regions
    
    def _initialize_neuron_positions(self):
        """Initialize 3D positions for all neurons using parallel processing."""
        print("Positioning neurons...")
        num_cpus = os.cpu_count() or 1
        chunk_size = min(10000, self.total_neurons // num_cpus)
        
        with mp.Pool(processes=num_cpus) as pool:
            for region in tqdm(self.regions.values(), desc="Positioning neurons by region"):
                for layer_config in region.layer_configs.values():
                    chunks = []
                    neurons_in_layer = layer_config.end_idx - layer_config.start_idx
                    
                    for chunk_start in range(0, neurons_in_layer, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, neurons_in_layer)
                        chunks.append((
                            layer_config.start_idx + chunk_start,
                            chunk_end - chunk_start,
                            region.center,
                            region.radius
                        ))
                    
                    results = list(pool.imap_unordered(_position_neurons_chunk, chunks))
                    
                    # Combine results
                    positions = np.zeros((neurons_in_layer, 3))
                    for start_idx, chunk_positions in results:
                        rel_start = start_idx - layer_config.start_idx
                        positions[rel_start:rel_start + len(chunk_positions)] = chunk_positions
                    
                    layer_config.positions = positions
    
    def _initialize_neurons(self):
        """Initialize neurons in parallel based on layer properties."""
        print("Initializing neurons...")
        num_cpus = os.cpu_count() or 1
        chunk_size = min(10000, self.total_neurons // num_cpus)
        
        with mp.Pool(processes=num_cpus) as pool:
            for region in tqdm(self.regions.values(), desc="Initializing neurons by region"):
                for layer_config in region.layer_configs.values():
                    chunks = []
                    for chunk_start in range(layer_config.start_idx, layer_config.end_idx, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, layer_config.end_idx)
                        chunks.append((chunk_start, chunk_end, layer_config.properties))
                    
                    # Process chunks in parallel
                    results = []
                    for chunk_neurons in pool.imap_unordered(_initialize_layer_neurons_chunk, chunks):
                        results.extend(chunk_neurons)
                    
                    # Sort by index and store
                    results.sort(key=lambda x: x[0])
                    layer_config.neurons = [n for _, n in results]
                    self.neurons.extend(layer_config.neurons)
    
    def _initialize_connections(self):
        """Initialize connections in parallel based on layer-specific connectivity rules."""
        print("Initializing connections...")
        num_cpus = os.cpu_count() or 1
        chunk_size = min(5000, self.total_neurons // num_cpus)
        
        with mp.Pool(processes=num_cpus) as pool:
            for region in tqdm(self.regions.values(), desc="Connecting regions"):
                for source_layer in region.layer_configs.values():
                    for target_layer in region.layer_configs.values():
                        self._connect_layers_parallel(source_layer, target_layer, chunk_size)
    
    def _connect_layers_parallel(self, source_layer: RegionLayerConfig, 
                               target_layer: RegionLayerConfig, chunk_size: int):
        """Create connections between layers using parallel processing."""
        num_cpus = os.cpu_count() or 1
        
        # Calculate distances between all neurons in the layers
        source_pos = source_layer.positions
        target_pos = target_layer.positions
        
        # Process in chunks to avoid memory issues
        with mp.Pool(processes=num_cpus) as pool:
            for i in range(0, len(source_layer.neurons), chunk_size):
                chunk_source = source_layer.neurons[i:i + chunk_size]
                chunk_source_pos = source_pos[i:i + chunk_size]
                
                # Calculate distances for this chunk
                distances = np.linalg.norm(
                    chunk_source_pos[:, np.newaxis] - target_pos, axis=2
                )
                
                # Get base probability from layer properties
                base_prob = source_layer.properties.connectivity_rules.get(
                    (chunk_source[0].type, target_layer.neurons[0].type), 0.0
                )
                
                # Process connections in parallel
                chunk_args = (chunk_source, target_layer.neurons, base_prob, distances)
                for connections in pool.imap_unordered(_process_connections_chunk, [chunk_args]):
                    for connection in connections:
                        # Store the connection object directly using source and target indices as key
                        self.connections[(connection.source_idx, connection.target_idx)] = connection
    
    def _validate_brain(self):
        """Validate the brain structure and report statistics."""
        print("\nBrain validation and statistics:")
        print(f"Total neurons: {len(self.neurons):,}")
        print(f"Total connections: {len(self.connections):,}")
        
        # Validate neuron indices
        invalid_neurons = [n for n in self.neurons if n.idx >= self.total_neurons]
        if invalid_neurons:
            print(f"Warning: Found {len(invalid_neurons)} neurons with invalid indices")
            
        # Validate connections
        invalid_connections = [(s, t) for (s, t) in self.connections.keys() 
                             if s >= self.total_neurons or t >= self.total_neurons]
        if invalid_connections:
            print(f"Warning: Found {len(invalid_connections)} connections with invalid indices")
    
    @property
    def total_neurons(self) -> int:
        """Get total number of neurons across all regions."""
        return sum(region.num_neurons for region in self.regions.values())
    
    def get_neuron(self, idx: int) -> Optional[AdvancedNeuron]:
        """Get neuron by index."""
        if 0 <= idx < len(self.neurons):
            return self.neurons[idx]
        return None
    
    def get_connection(self, source_idx: int, target_idx: int) -> Optional[AdvancedConnection]:
        """Get connection between two neurons."""
        return self.connections.get((source_idx, target_idx))
    
    def get_layer_neurons(self, region_name: str, layer: CorticalLayer) -> List[AdvancedNeuron]:
        """Get all neurons in a specific layer of a region."""
        region = self.regions.get(region_name)
        if region and layer in region.layer_configs:
            return region.layer_configs[layer].neurons
        return []