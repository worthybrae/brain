# simulation.py
import numpy as np
from typing import List
from dataclasses import dataclass
from enum import Enum
import scipy.signal as signal
from tqdm import tqdm
import concurrent.futures
from collections import deque

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
        
    def adjust_weights(self, current_rate: float, connections: List['AdvancedConnection']):
        """Scale synaptic weights to maintain target firing rate."""
        rate_error = self.target_rate - current_rate
        self.scaling_factor += self.adaptation_rate * rate_error
        self.scaling_factor = np.clip(self.scaling_factor, 0.5, 2.0)
        
        for conn in connections:
            conn.weight *= self.scaling_factor

class AdvancedConnection:
    """Implements biologically detailed synaptic connections."""
    def __init__(self, source_idx: int, target_idx: int, weight: float, delay: float):
        self.source_idx = source_idx
        self.target_idx = target_idx
        self.weight = weight
        self.delay = delay
        
        # STDP parameters
        self.A_plus = 0.005
        self.A_minus = 0.0025
        self.tau_plus = 20.0
        self.tau_minus = 40.0
        
        # Trace variables for STDP
        self.pre_trace = 0.0
        self.post_trace = 0.0
        
        # Synaptic state
        self.vesicle_pool = 1.0
        self.recovery_tau = 800.0  # ms
        self.facilitation = 1.0
        self.facilitation_tau = 100.0  # ms
        
        # Structural plasticity
        self.stability = 1.0
        self.pruning_threshold = 0.2
        
        # Receptor composition
        self.ampa_ratio = 0.8
        self.nmda_ratio = 0.2
        self.nmda_voltage_dependence = True
        
    def update_stdp(self, t: float, pre_spike: bool, post_spike: bool, dt: float):
        """Update STDP traces and weights."""
        # Decay traces
        self.pre_trace *= np.exp(-dt / self.tau_plus)
        self.post_trace *= np.exp(-dt / self.tau_minus)
        
        # Update traces and weights on spikes
        if pre_spike:
            dw = self.A_plus * self.post_trace
            self.weight += dw
            self.pre_trace += 1.0
            
        if post_spike:
            dw = -self.A_minus * self.pre_trace
            self.weight += dw
            self.post_trace += 1.0
            
        # Weight bounds
        self.weight = np.clip(self.weight, 0.0, 5.0)
        
    def compute_transmission(self, pre_spike: bool, dt: float, post_voltage: float) -> float:
        """Compute synaptic transmission including short-term plasticity."""
        if not pre_spike:
            # Recovery of vesicle pool
            self.vesicle_pool += dt * (1 - self.vesicle_pool) / self.recovery_tau
            # Decay of facilitation
            self.facilitation += dt * (1 - self.facilitation) / self.facilitation_tau
            return 0.0
            
        # Compute release probability
        P_release = self.facilitation * self.vesicle_pool
        
        # Update synaptic resources
        self.vesicle_pool *= (1 - P_release)
        self.facilitation += 0.1
        
        # Compute synaptic current components
        ampa_current = self.weight * self.ampa_ratio * P_release
        
        # NMDA current with voltage dependence
        mg_block = 1.0 / (1.0 + np.exp(-0.062 * post_voltage) * (1.0/3.57))
        nmda_current = self.weight * self.nmda_ratio * P_release * mg_block
        
        return ampa_current + nmda_current

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
                connection = self.brain.get_connection_info(source_idx, neuron.idx)
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