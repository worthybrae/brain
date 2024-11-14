# simulation.py
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import scipy.signal as signal
from tqdm import tqdm
import concurrent.futures
from collections import deque
from connection import AdvancedConnection

class NetworkState(Enum):
    RESTING = "resting"
    ACTIVE = "active"
    BURST = "burst"
    SLOW_WAVE = "slow_wave"
    REM = "rem"

@dataclass
class NetworkOscillation:
    """Tracks network-wide oscillatory activity."""
    frequency: float  # Hz
    power: float
    phase: float
    source_region: str

class SynapticHomeostasis:
    """Implements homeostatic plasticity mechanisms."""
    def __init__(self, target_rate: float = 5.0):
        self.target_rate = target_rate  # Target firing rate (Hz)
        self.scaling_factor = 1.0
        self.time_window = 1000.0  # ms
        self.adaptation_rate = 0.01
        
    def adjust_weights(self, current_rate: float, connections: Dict[Tuple[int, int], AdvancedConnection]):
        """Scale synaptic weights to maintain target firing rate."""
        rate_error = self.target_rate - current_rate
        self.scaling_factor += self.adaptation_rate * rate_error
        self.scaling_factor = np.clip(self.scaling_factor, 0.5, 2.0)
        
        # Iterate over the values (AdvancedConnection objects) of the connections dictionary
        for connection in connections.values():
            connection.weight *= self.scaling_factor

class AdvancedSimulation:
    """Implements biologically detailed neural network simulation."""
    def __init__(self, brain, duration: float = 1000.0, dt: float = 0.1):
        self.brain = brain
        self.duration = duration
        self.dt = dt
        self.time_steps = int(duration / dt)
        self.time = np.arange(0, duration, dt)
        
        # Network state tracking
        self.state = NetworkState.RESTING
        self.oscillations: List[NetworkOscillation] = []
        self.activity_history = deque(maxlen=1000)
        
        # Homeostatic mechanisms
        self.homeostasis = SynapticHomeostasis()
        
        # Background activity
        self.noise_amplitude = 0.1
        self.noise_correlation = 0.1
        
        # Recording settings
        self.record_voltage = True
        self.record_calcium = True
        self.record_synaptic = True
        
        # Initialize recording arrays
        self.voltage_record = np.zeros((self.time_steps, len(brain.neurons)))
        self.calcium_record = np.zeros((self.time_steps, len(brain.neurons)))
        self.spike_record = np.zeros((self.time_steps, len(brain.neurons)), dtype=bool)
        
        # Network analysis
        self.firing_rates = np.zeros(len(brain.neurons))
        self.synchrony_index = 0.0
        self.phase_coherence = 0.0
        
    def generate_background_activity(self) -> np.ndarray:
        """Generate correlated background noise for all neurons."""
        shared_noise = np.random.normal(0, self.noise_amplitude, self.time_steps)
        individual_noise = np.random.normal(0, self.noise_amplitude, 
                                         (self.time_steps, len(self.brain.neurons)))
        
        # Mix shared and individual noise based on correlation
        total_noise = (np.sqrt(self.noise_correlation) * shared_noise[:, np.newaxis] +
                      np.sqrt(1 - self.noise_correlation) * individual_noise)
        
        return total_noise
        
    def detect_network_state(self):
        """Analyze network activity to determine current state."""
        recent_activity = np.array(self.activity_history)
        if len(recent_activity) < 100:
            return
            
        # Compute power spectrum
        freqs, psd = signal.welch(recent_activity.mean(axis=1), fs=1000/self.dt)
        
        # Detect dominant oscillations
        peaks, properties = signal.find_peaks(psd, height=np.mean(psd)*2)
        
        self.oscillations = []
        for peak, height in zip(peaks, properties['peak_heights']):
            self.oscillations.append(NetworkOscillation(
                frequency=freqs[peak],
                power=height,
                phase=np.angle(signal.hilbert(recent_activity.mean(axis=1)))[0],
                source_region='unknown'
            ))
            
        # Determine network state based on oscillations
        if len(self.oscillations) == 0:
            self.state = NetworkState.RESTING
        elif any(0.5 <= osc.frequency <= 4 for osc in self.oscillations):
            self.state = NetworkState.SLOW_WAVE
        elif any(4 <= osc.frequency <= 8 for osc in self.oscillations):
            self.state = NetworkState.BURST
        elif any(30 <= osc.frequency <= 80 for osc in self.oscillations):
            self.state = NetworkState.ACTIVE
            
    def update_network_metrics(self, t_idx: int):
        """Update various network analysis metrics."""
        # Update firing rates
        window = 100  # ms
        if t_idx >= window:
            self.firing_rates = np.mean(self.spike_record[t_idx-window:t_idx], axis=0) * 1000/window
            
        # Compute synchrony index (normalized cross-correlation)
        if t_idx >= 1000:
            recent_spikes = self.spike_record[t_idx-1000:t_idx]
            coincident_spikes = np.sum(np.corrcoef(recent_spikes.T))
            self.synchrony_index = coincident_spikes / (len(self.brain.neurons) ** 2)
            
        # Phase coherence using Hilbert transform
        if t_idx >= 1000:
            analytic_signal = signal.hilbert(self.voltage_record[t_idx-1000:t_idx])
            phases = np.angle(analytic_signal)
            self.phase_coherence = np.abs(np.mean(np.exp(1j * phases)))
            
    def run(self):
        """Execute the simulation with parallel neuron updates."""
        background_noise = self.generate_background_activity()
        
        for t_idx, t in enumerate(tqdm(self.time, desc="Simulating")):
            # Update each neuron in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for nrn_idx, neuron in enumerate(self.brain.neurons):
                    futures.append(executor.submit(
                        self._update_neuron,
                        neuron,
                        t,
                        t_idx,
                        background_noise[t_idx, nrn_idx]
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
            if t_idx % 1000 == 0:  # Every second
                self.homeostasis.adjust_weights(
                    np.mean(self.firing_rates),
                    self.brain.connections
                )
                
    def _update_neuron(self, neuron, t: float, t_idx: int, noise: float) -> tuple:
        """Update single neuron and its synapses."""
        # Collect synaptic inputs
        for source_idx, synapse in neuron.synapses.items():
            # Find presynaptic spikes
            if t_idx > 0 and self.spike_record[t_idx-1, source_idx]:
                connection = self.brain.get_connection(source_idx, neuron.idx)
                if connection:
                    # Compute synaptic transmission
                    psp = connection.compute_transmission(True, self.dt, neuron.v)
                    neuron.receive_input(source_idx, psp, t)
                    
                    # Update STDP
                    connection.update_stdp(t, True, False, self.dt)
        
        # Add background noise
        neuron.receive_input(-1, noise, t)
        
        # Integrate neuron dynamics
        neuron.integrate(t, self.dt)
        
        # Return state variables for recording
        return neuron.idx, neuron.v, neuron.Ca, neuron.spike