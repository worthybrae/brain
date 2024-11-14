import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
import scipy.sparse as sparse
from scipy.spatial import distance

@dataclass
class BrainRegion:
    """Represents a brain region's spatial and neural properties"""
    name: str
    num_neurons: int
    center: Tuple[float, float, float]  # (x, y, z) center of region
    radius: float  # approximate radius of region
    start_idx: int = 0  # Starting index in the global neuron array
    end_idx: int = 0    # Ending index in the global neuron array

class NeuralConnection:
    """Represents a connection between neurons with weight and girth"""
    def __init__(self, weight: float, girth: float):
        self.weight = weight  # Connection strength
        self.girth = girth    # Physical width of connection (in micrometers)
        
    def __float__(self):
        # Allow backwards compatibility with weight-only calculations
        return float(self.weight)

class DigitalBrain:
    def __init__(self):
        # Initialize class attributes first
        self.base_girth = 0.5  # Base girth in micrometers
        self.girth_variance = 0.1  # Variance in girth
        self.connection_properties = {}  # Initialize the dictionary first
        
        # Define brain regions with spatial information (coordinates in mm)
        self.regions = {
            # Visual Processing - posterior regions
            'v1': BrainRegion('v1', 5_000_000, (0, -80, 0), 20),    
            'v2': BrainRegion('v2', 3_000_000, (0, -70, 0), 15),    
            'v4': BrainRegion('v4', 2_000_000, (30, -60, -5), 12),  
            'it': BrainRegion('it', 3_000_000, (40, -50, -10), 15), 
            
            # Auditory Processing - temporal regions
            'a1': BrainRegion('a1', 2_000_000, (50, -20, 5), 10),   
            'a2': BrainRegion('a2', 3_000_000, (55, -15, 5), 12),   
            
            # Integration & Higher Processing
            'pfc': BrainRegion('pfc', 6_000_000, (30, 50, 0), 25),  
            'hc': BrainRegion('hc', 4_000_000, (25, -20, -20), 15), 
        }
        
        # Calculate total neurons
        self.total_neurons = sum(r.num_neurons for r in self.regions.values())
        
        # Initialize components in order
        self._assign_neuron_indices()
        self.neuron_positions = self._initialize_neuron_positions()
        self.neuron_states = np.zeros(self.total_neurons)
        self.connections = self._initialize_connections()

    def _assign_neuron_indices(self):
        """Assign start and end indices for each region"""
        current_idx = 0
        for region in self.regions.values():
            region.start_idx = current_idx
            region.end_idx = current_idx + region.num_neurons
            current_idx = region.end_idx

    def _initialize_neuron_positions(self) -> np.ndarray:
        """Initialize 3D positions for all neurons"""
        positions = np.zeros((self.total_neurons, 3))
        
        for region in self.regions.values():
            n_neurons = region.end_idx - region.start_idx
            
            # Generate random angles and radii
            theta = np.random.uniform(0, 2*np.pi, n_neurons)
            phi = np.arccos(np.random.uniform(-1, 1, n_neurons))
            r = region.radius * np.cbrt(np.random.uniform(0, 1, n_neurons))
            
            # Convert to Cartesian coordinates
            x = r * np.sin(phi) * np.cos(theta) + region.center[0]
            y = r * np.sin(phi) * np.sin(theta) + region.center[1]
            z = r * np.cos(phi) + region.center[2]
            
            positions[region.start_idx:region.end_idx] = np.column_stack((x, y, z))
        
        return positions

    def _calculate_connection_properties(self, distance: float, base_prob: float) -> Tuple[float, float]:
        """Calculate connection probability and girth based on distance"""
        # Probability decreases with distance
        probability = base_prob * np.exp(-distance / 50.0)
        
        # Girth is influenced by distance (connections to nearby neurons are generally thicker)
        girth = self.base_girth * np.exp(-distance / 100.0) + \
                np.random.normal(0, self.girth_variance)
        
        # Ensure minimum girth
        girth = max(0.1, girth)
        
        return probability, girth

    def _initialize_connections(self) -> sparse.csr_matrix:
        """Initialize sparse connectivity matrix with distance-based probabilities and girths"""
        connections = sparse.lil_matrix((self.total_neurons, self.total_neurons))
        
        # Define pathways with base probabilities
        pathways = {
            ('v1', 'v2'): 0.1,
            ('v2', 'v4'): 0.1,
            ('v4', 'it'): 0.1,
            ('it', 'pfc'): 0.05,
            ('a1', 'a2'): 0.1,
            ('a2', 'pfc'): 0.05,
            ('pfc', 'hc'): 0.05,
            ('hc', 'pfc'): 0.05,
        }
        
        # Use smaller number of neurons for testing
        sample_size = 100  # Reduce this number for faster initialization
        
        for (source_name, target_name), base_prob in pathways.items():
            source = self.regions[source_name]
            target = self.regions[target_name]
            
            # Sample subset of neurons for efficiency
            source_samples = np.random.choice(
                range(source.start_idx, source.end_idx),
                size=min(sample_size, source.end_idx - source.start_idx),
                replace=False
            )
            
            for source_idx in source_samples:
                source_pos = self.neuron_positions[source_idx]
                target_positions = self.neuron_positions[target.start_idx:target.end_idx]
                
                # Calculate distances to a subset of target neurons
                target_samples = np.random.choice(
                    range(len(target_positions)),
                    size=min(sample_size, len(target_positions)),
                    replace=False
                )
                
                distances = distance.cdist([source_pos], target_positions[target_samples])[0]
                
                # Calculate connection properties for each potential connection
                for i, dist in enumerate(distances):
                    target_idx = target.start_idx + target_samples[i]
                    prob, girth = self._calculate_connection_properties(dist, base_prob)
                    
                    if np.random.random() < prob:
                        # Generate weight (correlated with girth)
                        weight = 0.5 + 0.1 * girth + np.random.normal(0, 0.1)
                        
                        # Store connection
                        connections[source_idx, target_idx] = weight
                        self.connection_properties[(source_idx, target_idx)] = \
                            NeuralConnection(weight, girth)
        
        return connections.tocsr()

    def get_connection_info(self, source_idx: int, target_idx: int) -> NeuralConnection:
        """Get the connection properties between two neurons"""
        return self.connection_properties.get((source_idx, target_idx))

    def strengthen_connection(self, source_idx: int, target_idx: int, amount: float = 0.1):
        """Strengthen a connection by increasing both weight and girth"""
        if (source_idx, target_idx) in self.connection_properties:
            conn = self.connection_properties[(source_idx, target_idx)]
            conn.weight += amount
            conn.girth += amount * 0.1  # Girth changes more slowly than weight

if __name__ == "__main__":
    print("Initializing brain...")
    brain = DigitalBrain()
    print("Brain initialized!")
    
    # Example: Get connection properties between two neurons
    v1_neuron = brain.regions['v1'].start_idx + 50
    v2_neuron = brain.regions['v2'].start_idx + 50
    
    connection = brain.get_connection_info(v1_neuron, v2_neuron)
    if connection:
        print(f"Connection weight: {connection.weight:.3f}")
        print(f"Connection girth: {connection.girth:.3f} micrometers")
    else:
        print("No connection found between these neurons")