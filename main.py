import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import scipy.sparse as sparse
from scipy.spatial import distance
from tqdm import tqdm
from logger import BrainLogger
import time
import multiprocessing as mp
from functools import partial
import os

# Initialize logger at module level
logger = BrainLogger.get_logger()

@dataclass
class BrainRegion:
    """Represents a brain region's spatial and neural properties"""
    name: str
    num_neurons: int
    center: Tuple[float, float, float]
    radius: float
    start_idx: int = 0
    end_idx: int = 0

class NeuralConnection:
    def __init__(self, weight: float, girth: float):
        self.weight = weight
        self.girth = girth
        
    def __float__(self):
        return float(self.weight)

def _position_neurons_chunk(args):
    """Worker function to position neurons in parallel"""
    try:
        start_idx, chunk_size, center, radius = args
        chunk_size_actual = chunk_size
        
        theta = np.random.uniform(0, 2*np.pi, chunk_size_actual)
        phi = np.arccos(np.random.uniform(-1, 1, chunk_size_actual))
        r = radius * np.cbrt(np.random.uniform(0, 1, chunk_size_actual))
        
        x = r * np.sin(phi) * np.cos(theta) + center[0]
        y = r * np.sin(phi) * np.sin(theta) + center[1]
        z = r * np.cos(phi) + center[2]
        
        return start_idx, np.column_stack((x, y, z))
    except Exception as e:
        logger.error(f"Error in _position_neurons_chunk: {str(e)}")
        return None

def _process_connection_chunk(args):
    """Worker function to process connections in parallel"""
    try:
        source_samples, target_samples, source_pos, target_positions, base_prob, base_girth, girth_variance = args
        connections = []
        connection_properties = {}
        
        distances = distance.cdist([source_pos], target_positions)[0]
        
        for i, dist in enumerate(distances):
            target_idx = target_samples[i]
            
            probability = base_prob * np.exp(-dist / 50.0)
            
            if np.random.random() < probability:
                girth = base_girth * np.exp(-dist / 100.0) + np.random.normal(0, girth_variance)
                girth = max(0.1, girth)
                weight = 0.5 + 0.1 * girth + np.random.normal(0, 0.1)
                
                connections.append((source_samples, target_idx, weight))
                connection_properties[(source_samples, target_idx)] = NeuralConnection(weight, girth)
        
        return connections, connection_properties
    except Exception as e:
        logger.error(f"Error in _process_connection_chunk: {str(e)}")
        return None

class DigitalBrain:
    def __init__(self):
        logger.info("Initializing Digital Brain...")
        start_time = time.time()
        
        self.base_girth = 0.5
        self.girth_variance = 0.1
        self.connection_properties = {}
        
        # Reduced number of neurons for testing
        scale_factor = 0.01  # 1% of original size
        self.regions = {
            'v1': BrainRegion('v1', int(2_000_000_000 * scale_factor), (0, -80, 0), 20),
            'v2': BrainRegion('v2', int(1_000_000_000 * scale_factor), (0, -70, 0), 15),
            'v4': BrainRegion('v4', int(1_000_000_000 * scale_factor), (30, -60, -5), 12),
            'it': BrainRegion('it', int(2_000_000_000 * scale_factor), (40, -50, -10), 15),
            'a1': BrainRegion('a1', int(1_000_000_000 * scale_factor), (50, -20, 5), 10),
            'a2': BrainRegion('a2', int(1_000_000_000 * scale_factor), (55, -15, 5), 12),
            'pfc': BrainRegion('pfc', int(4_000_000_000 * scale_factor), (30, 50, 0), 25),
            'hc': BrainRegion('hc', int(2_000_000_000 * scale_factor), (25, -20, -20), 15),
        }
        
        self.total_neurons = sum(r.num_neurons for r in self.regions.values())
        logger.info(f"Total neurons: {self.total_neurons:,}")
        
        self._assign_neuron_indices()
        self.neuron_positions = self._initialize_neuron_positions()
        self.connections = self._initialize_connections()
        
        end_time = time.time()
        logger.info(f"Brain initialization completed in {end_time - start_time:.2f} seconds")

    def _initialize_neuron_positions(self) -> np.ndarray:
        """Initialize 3D positions for all neurons using parallel processing"""
        logger.info("Initializing neuron positions...")
        
        positions = np.zeros((self.total_neurons, 3), dtype=np.float32)
        chunk_size = min(100_000, self.total_neurons // (os.cpu_count() or 1))  # Smaller chunks
        
        with mp.Pool(processes=os.cpu_count()) as pool:
            with tqdm(total=self.total_neurons, desc="Positioning neurons", unit="neurons") as pbar:
                for region in self.regions.values():
                    chunks = []
                    
                    for chunk_start in range(region.start_idx, region.end_idx, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, region.end_idx)
                        chunk_size_actual = chunk_end - chunk_start
                        chunks.append((chunk_start, chunk_size_actual, region.center, region.radius))
                    
                    try:
                        for result in pool.imap_unordered(_position_neurons_chunk, chunks):
                            if result is not None:
                                start_idx, chunk_positions = result
                                end_idx = start_idx + len(chunk_positions)
                                positions[start_idx:end_idx] = chunk_positions
                                pbar.update(len(chunk_positions))
                    except Exception as e:
                        logger.error(f"Error in position initialization: {str(e)}")
                        pool.terminate()
                        raise
        
        return positions

    def _initialize_connections(self) -> sparse.csr_matrix:
        """Initialize sparse connectivity matrix using parallel processing"""
        logger.info("Initializing neural connections...")
        connections = sparse.lil_matrix((self.total_neurons, self.total_neurons), dtype=np.float32)
        
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
        
        # Reduced sample size
        sample_size = 1_000
        chunk_size = 100
        
        total_ops = len(pathways) * sample_size
        
        with mp.Pool(processes=os.cpu_count()) as pool:
            with tqdm(total=total_ops, desc="Creating connections", unit="connections") as pbar:
                for (source_name, target_name), base_prob in pathways.items():
                    source = self.regions[source_name]
                    target = self.regions[target_name]
                    
                    source_samples = np.random.choice(
                        range(source.start_idx, source.end_idx),
                        size=min(sample_size, source.end_idx - source.start_idx),
                        replace=False
                    )
                    
                    try:
                        for i in range(0, len(source_samples), chunk_size):
                            chunk_source_samples = source_samples[i:i + chunk_size]
                            chunks = []
                            
                            for source_idx in chunk_source_samples:
                                target_samples = np.random.choice(
                                    range(target.start_idx, target.end_idx),
                                    size=min(sample_size, target.end_idx - target.start_idx),
                                    replace=False
                                )
                                chunks.append((
                                    source_idx,
                                    target_samples,
                                    self.neuron_positions[source_idx],
                                    self.neuron_positions[target_samples],
                                    base_prob,
                                    self.base_girth,
                                    self.girth_variance
                                ))
                            
                            for result in pool.imap_unordered(_process_connection_chunk, chunks):
                                if result is not None:
                                    chunk_connections, chunk_properties = result
                                    for source_idx, target_idx, weight in chunk_connections:
                                        connections[source_idx, target_idx] = weight
                                    self.connection_properties.update(chunk_properties)
                                    pbar.update(len(chunk_properties))
                    except Exception as e:
                        logger.error(f"Error in connection initialization: {str(e)}")
                        pool.terminate()
                        raise
        
        logger.info(f"Total connections created: {len(self.connection_properties):,}")
        return connections.tocsr()

    def _assign_neuron_indices(self):
        """Assign start and end indices for each region"""
        current_idx = 0
        for region in self.regions.values():
            region.start_idx = current_idx
            region.end_idx = current_idx + region.num_neurons
            current_idx = region.end_idx

    def get_connection_info(self, source_idx: int, target_idx: int) -> NeuralConnection:
        """Get the connection properties between two neurons"""
        return self.connection_properties.get((source_idx, target_idx))

    def strengthen_connection(self, source_idx: int, target_idx: int, amount: float = 0.1):
        """Strengthen a connection by increasing both weight and girth"""
        if (source_idx, target_idx) in self.connection_properties:
            conn = self.connection_properties[(source_idx, target_idx)]
            conn.weight += amount
            conn.girth += amount * 0.1
            logger.debug(f"Strengthened connection ({source_idx}, {target_idx}): "
                        f"weight={conn.weight:.3f}, girth={conn.girth:.3f}")

if __name__ == "__main__":
    logger.info("Starting brain simulation...")
    try:
        brain = DigitalBrain()
        
        # Test connection properties
        v1_neuron = brain.regions['v1'].start_idx + 50
        v2_neuron = brain.regions['v2'].start_idx + 50
        
        connection = brain.get_connection_info(v1_neuron, v2_neuron)
        if connection:
            logger.info(f"Test connection found:")
            logger.info(f"  Weight: {connection.weight:.3f}")
            logger.info(f"  Girth: {connection.girth:.3f} micrometers")
        else:
            logger.info("No connection found between test neurons")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise