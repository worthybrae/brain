# checkpointing.py
import json
import numpy as np
import pickle
import concurrent.futures
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from collections import deque
from tqdm import tqdm

# Import required classes from simulation.py
from simulation import (
    AdvancedSimulation,
    NetworkState,
    NetworkOscillation
)

# Import required classes from digital_brain.py
from digital_brain import ModularDigitalBrain

class CheckpointManager:
    """Manages saving and loading simulation checkpoints."""
    
    def __init__(self, save_dir: str = "checkpoints", checkpoint_frequency: int = 1000):
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to store checkpoints
            checkpoint_frequency: Save checkpoint every N timesteps
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_frequency = checkpoint_frequency
        self.last_checkpoint = None
        
    def _get_checkpoint_path(self, simulation_id: str, timestep: int) -> Path:
        """Generate checkpoint file path."""
        return self.save_dir / f"checkpoint_{simulation_id}_{timestep}.pkl"
        
    def _get_metadata_path(self, simulation_id: str) -> Path:
        """Generate metadata file path."""
        return self.save_dir / f"metadata_{simulation_id}.json"
    
    def save_checkpoint(self, simulation: 'ResumableSimulation', timestep: int) -> None:
        """
        Save simulation state to checkpoint.
        
        Args:
            simulation: ResumableSimulation instance
            timestep: Current simulation timestep
        """
        # Only save at specified frequency
        if timestep % self.checkpoint_frequency != 0:
            return
            
        simulation_id = self._get_simulation_id(simulation)
        
        # Save simulation state
        checkpoint_data = {
            'timestep': timestep,
            'duration': simulation.duration,
            'dt': simulation.dt,
            'state': simulation.state.value,
            'time': simulation.time.tolist(),
            'voltage_record': simulation.voltage_record[:timestep+1].tolist(),
            'calcium_record': simulation.calcium_record[:timestep+1].tolist(),
            'spike_record': simulation.spike_record[:timestep+1].tolist(),
            'oscillations': [asdict(osc) for osc in simulation.oscillations],
            'activity_history': list(simulation.activity_history),
            'firing_rates': simulation.firing_rates.tolist(),
            'synchrony_index': simulation.synchrony_index,
            'phase_coherence': simulation.phase_coherence,
            
            # Save homeostasis state
            'homeostasis': {
                'scaling_factor': simulation.homeostasis.scaling_factor,
                'target_rate': simulation.homeostasis.target_rate,
                'adaptation_rate': simulation.homeostasis.adaptation_rate
            }
        }
        
        # Save brain state
        brain_state = {
            'neurons': [
                {
                    'idx': n.idx,
                    'v': n.v,
                    'Ca': n.Ca,
                    'spike': n.spike,
                    'last_spike_time': n.last_spike_time,
                    'w': n.w,  # Adaptation current
                    'spike_threshold': n.spike_threshold,
                    'atp': n.atp,
                    'energy_usage': n.energy_usage,
                    # Save synapse states
                    'synapses': {
                        src_idx: {
                            'g': syn.g,
                            'depression': syn.depression,
                            'facilitation': syn.facilitation,
                            'last_spike_time': syn.last_spike_time
                        } for src_idx, syn in n.synapses.items()
                    }
                } for n in simulation.brain.neurons
            ],
            'connections': [
                {
                    'source_idx': conn.source_idx,
                    'target_idx': conn.target_idx,
                    'weight': conn.weight,
                    'pre_trace': conn.pre_trace,
                    'post_trace': conn.post_trace,
                    'vesicle_pool': conn.vesicle_pool,
                    'facilitation': conn.facilitation,
                    'stability': conn.stability,
                    'last_spike_time': conn.last_spike_time
                } for conn in simulation.brain.connections.values()
            ]
        }
        checkpoint_data['brain_state'] = brain_state
        
        # Save checkpoint
        checkpoint_path = self._get_checkpoint_path(simulation_id, timestep)
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # Update metadata
        metadata = {
            'simulation_id': simulation_id,
            'last_timestep': timestep,
            'checkpoint_path': str(checkpoint_path),
            'timestamp': datetime.now().isoformat(),
            'duration': simulation.duration,
            'dt': simulation.dt,
            'scale_factor': simulation.brain.scale_factor,
            'total_neurons': len(simulation.brain.neurons),
            'total_connections': len(simulation.brain.connections)
        }
        
        metadata_path = self._get_metadata_path(simulation_id)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.last_checkpoint = checkpoint_path
    
    def load_latest_checkpoint(self, simulation_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Load most recent checkpoint for given simulation.
        
        Returns:
            Tuple of (checkpoint_data, metadata) if found, else (None, None)
        """
        metadata_path = self._get_metadata_path(simulation_id)
        if not metadata_path.exists():
            return None, None
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        checkpoint_path = Path(metadata['checkpoint_path'])
        if not checkpoint_path.exists():
            return None, None
            
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
            
        return checkpoint_data, metadata
    
    def _get_simulation_id(self, simulation: 'ResumableSimulation') -> str:
        """Generate unique simulation identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"sim_{timestamp}_{simulation.brain.scale_factor}"
    
    def clean_old_checkpoints(self, keep_last_n: int = 5) -> None:
        """Remove old checkpoints, keeping only the N most recent."""
        checkpoints = sorted(self.save_dir.glob("checkpoint_*.pkl"))
        if len(checkpoints) > keep_last_n:
            for checkpoint in checkpoints[:-keep_last_n]:
                checkpoint.unlink()
                # Also remove corresponding metadata
                simulation_id = checkpoint.stem.split('_')[1]
                metadata_path = self._get_metadata_path(simulation_id)
                if metadata_path.exists():
                    metadata_path.unlink()

class SimulationRecovery:
    """Handles simulation recovery from checkpoints."""
    
    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager
    
    def restore_simulation(self, simulation: 'ResumableSimulation', checkpoint_data: Dict[str, Any]) -> None:
        """
        Restore simulation state from checkpoint data.
        
        Args:
            simulation: ResumableSimulation instance to restore
            checkpoint_data: Checkpoint data dictionary
        """
        # Restore simulation parameters
        simulation.time = np.array(checkpoint_data['time'])
        simulation.state = NetworkState(checkpoint_data['state'])
        
        # Restore recorded data
        last_timestep = checkpoint_data['timestep']
        simulation.voltage_record[:last_timestep+1] = np.array(checkpoint_data['voltage_record'])
        simulation.calcium_record[:last_timestep+1] = np.array(checkpoint_data['calcium_record'])
        simulation.spike_record[:last_timestep+1] = np.array(checkpoint_data['spike_record'])
        
        # Restore oscillations
        simulation.oscillations = [
            NetworkOscillation(**osc_data) 
            for osc_data in checkpoint_data['oscillations']
        ]
        
        # Restore other metrics
        simulation.activity_history = deque(checkpoint_data['activity_history'], 
                                         maxlen=simulation.activity_history.maxlen)
        simulation.firing_rates = np.array(checkpoint_data['firing_rates'])
        simulation.synchrony_index = checkpoint_data['synchrony_index']
        simulation.phase_coherence = checkpoint_data['phase_coherence']
        
        # Restore homeostasis state
        homeostasis_state = checkpoint_data['homeostasis']
        simulation.homeostasis.scaling_factor = homeostasis_state['scaling_factor']
        simulation.homeostasis.target_rate = homeostasis_state['target_rate']
        simulation.homeostasis.adaptation_rate = homeostasis_state['adaptation_rate']
        
        # Restore brain state
        brain_state = checkpoint_data['brain_state']
        
        # Restore neuron states
        for neuron_data in brain_state['neurons']:
            neuron = simulation.brain.neurons[neuron_data['idx']]
            neuron.v = neuron_data['v']
            neuron.Ca = neuron_data['Ca']
            neuron.spike = neuron_data['spike']
            neuron.last_spike_time = neuron_data['last_spike_time']
            neuron.w = neuron_data['w']
            neuron.spike_threshold = neuron_data['spike_threshold']
            neuron.atp = neuron_data['atp']
            neuron.energy_usage = neuron_data['energy_usage']
            
            # Restore synapse states
            for src_idx, syn_data in neuron_data['synapses'].items():
                if src_idx in neuron.synapses:
                    syn = neuron.synapses[src_idx]
                    syn.g = syn_data['g']
                    syn.depression = syn_data['depression']
                    syn.facilitation = syn_data['facilitation']
                    syn.last_spike_time = syn_data['last_spike_time']
        
        # Restore connection states
        for conn_data in brain_state['connections']:
            key = (conn_data['source_idx'], conn_data['target_idx'])
            if key in simulation.brain.connections:
                conn = simulation.brain.connections[key]
                conn.weight = conn_data['weight']
                conn.pre_trace = conn_data['pre_trace']
                conn.post_trace = conn_data['post_trace']
                conn.vesicle_pool = conn_data['vesicle_pool']
                conn.facilitation = conn_data['facilitation']
                conn.stability = conn_data['stability']
                conn.last_spike_time = conn_data['last_spike_time']
        
        print(f"Restored simulation state from timestep {last_timestep}")

class ResumableSimulation(AdvancedSimulation):
    """Enhanced simulation class with checkpointing support."""
    
    def __init__(self, brain: ModularDigitalBrain, duration: float = 1000.0, dt: float = 0.1,
                 checkpoint_frequency: int = 1000):
        super().__init__(brain, duration, dt)
        self.checkpoint_manager = CheckpointManager(checkpoint_frequency=checkpoint_frequency)
        self.recovery = SimulationRecovery(self.checkpoint_manager)
        
        # Pre-generate background noise for entire simulation
        self.background_noise = self.generate_background_activity()
        
    def run(self, resume_id: Optional[str] = None):
        """
        Run simulation with checkpointing and optional resume.
        
        Args:
            resume_id: Optional simulation ID to resume from
        """
        start_timestep = 0
        
        # Try to resume from checkpoint
        if resume_id:
            checkpoint_data, metadata = self.checkpoint_manager.load_latest_checkpoint(resume_id)
            if checkpoint_data:
                self.recovery.restore_simulation(self, checkpoint_data)
                start_timestep = checkpoint_data['timestep'] + 1
                print(f"Resuming simulation from timestep {start_timestep}")
            else:
                print(f"No checkpoint found for simulation {resume_id}, starting fresh")
        
        try:
            # Run simulation loop with progress bar
            total_steps = self.time_steps - start_timestep
            with tqdm(total=total_steps, desc="Simulating", unit="steps") as pbar:
                for t_idx in range(start_timestep, self.time_steps):
                    self._update_timestep(t_idx)
                    self.checkpoint_manager.save_checkpoint(self, t_idx)
                    pbar.update(1)
                    
        except Exception as e:
            print(f"Simulation interrupted at timestep {t_idx}: {str(e)}")
            # Save final checkpoint on error
            self.checkpoint_manager.save_checkpoint(self, t_idx)
            raise
        
        finally:
            # Clean up old checkpoints
            self.checkpoint_manager.clean_old_checkpoints()
    
    def _update_timestep(self, t_idx: int):
        """Update single simulation timestep."""
        t = self.time[t_idx]
        
        # Update neurons
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for nrn_idx, neuron in enumerate(self.brain.neurons):
                futures.append(executor.submit(
                    self._update_neuron,
                    neuron,
                    t,
                    t_idx,
                    self.background_noise[t_idx, nrn_idx]
                ))
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                nrn_idx, v, ca, spike = future.result()
                self.voltage_record[t_idx, nrn_idx] = v
                self.calcium_record[t_idx, nrn_idx] = ca
                self.spike_record[t_idx, nrn_idx] = spike
        
        # Update network state and metrics
        self.activity_history.append(self.spike_record[t_idx])
        self.detect_network_state()
        self.update_network_metrics(t_idx)
        
        # Apply homeostatic plasticity
        if t_idx % 1000 == 0:
            self.homeostasis.adjust_weights(
                np.mean(self.firing_rates),
                self.brain.connections
            )