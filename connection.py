# connection.py
import numpy as np
from enum import Enum
from dataclasses import dataclass

class SynapseType(Enum):
    """Types of synaptic connections."""
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MODULATORY = "modulatory"
    ELECTRICAL = "electrical"  # Gap junctions

@dataclass
class STDPParameters:
    """Spike-timing-dependent plasticity parameters."""
    A_plus: float = 0.005    # LTP amplitude
    A_minus: float = 0.0025  # LTD amplitude
    tau_plus: float = 20.0   # LTP time constant (ms)
    tau_minus: float = 40.0  # LTD time constant (ms)
    w_max: float = 5.0       # Maximum weight
    w_min: float = 0.0       # Minimum weight

class AdvancedConnection:
    """Implements biologically detailed synaptic connections with plasticity."""
    
    def __init__(self, source_idx: int, target_idx: int, weight: float, delay: float,
                 synapse_type: SynapseType = SynapseType.EXCITATORY):
        self.source_idx = source_idx
        self.target_idx = target_idx
        self.weight = weight
        self.delay = delay  # Synaptic delay in ms
        self.synapse_type = synapse_type
        
        # STDP parameters
        self.stdp_params = STDPParameters()
        self.pre_trace = 0.0   # Presynaptic trace for STDP
        self.post_trace = 0.0  # Postsynaptic trace for STDP
        
        # Short-term plasticity (STP) parameters
        self.vesicle_pool = 1.0      # Available synaptic resources
        self.recovery_tau = 800.0     # Recovery time constant (ms)
        self.facilitation = 1.0       # Facilitation factor
        self.facilitation_tau = 100.0 # Facilitation decay time constant (ms)
        self.depression_factor = 0.2  # Depression factor per spike
        
        # Structural plasticity parameters
        self.stability = 1.0          # Synapse stability factor
        self.pruning_threshold = 0.2  # Minimum weight for survival
        self.growth_factor = 0.1      # Rate of weight increase when stable
        
        # Receptor composition
        self.ampa_ratio = 0.8   # Proportion of AMPA receptors
        self.nmda_ratio = 0.2   # Proportion of NMDA receptors
        self.gabaa_ratio = 0.0  # Proportion of GABAA receptors
        self.gabab_ratio = 0.0  # Proportion of GABAB receptors
        
        # Set receptor ratios based on synapse type
        if synapse_type == SynapseType.INHIBITORY:
            self.ampa_ratio = 0.0
            self.nmda_ratio = 0.0
            self.gabaa_ratio = 0.8
            self.gabab_ratio = 0.2
            
        # Activity history for metaplasticity
        self.activation_history = []
        self.max_history_length = 1000
        
        # Last update time
        self.last_spike_time = -1000.0
        
    def update_stdp(self, t: float, pre_spike: bool, post_spike: bool, dt: float):
        """Update STDP traces and weights."""
        # Decay traces
        self.pre_trace *= np.exp(-dt / self.stdp_params.tau_plus)
        self.post_trace *= np.exp(-dt / self.stdp_params.tau_minus)
        
        # Update traces and weights on spikes
        if pre_spike:
            dw = self.stdp_params.A_plus * self.post_trace
            self.weight += dw
            self.pre_trace += 1.0
            
        if post_spike:
            dw = -self.stdp_params.A_minus * self.pre_trace
            self.weight += dw
            self.post_trace += 1.0
            
        # Weight bounds
        self.weight = np.clip(self.weight, 
                            self.stdp_params.w_min, 
                            self.stdp_params.w_max)
        
    def compute_transmission(self, pre_spike: bool, dt: float, 
                           post_voltage: float = 0.0) -> float:
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
        self.facilitation += self.depression_factor
        
        # Record spike time for future reference
        self.last_spike_time = dt
        
        # Compute receptor-specific currents
        current = 0.0
        
        if self.synapse_type == SynapseType.EXCITATORY:
            # AMPA current (voltage-independent)
            current += self.weight * self.ampa_ratio * P_release
            
            # NMDA current (voltage-dependent)
            if self.nmda_ratio > 0:
                # Magnesium block factor
                mg_block = 1.0 / (1.0 + np.exp(-0.062 * post_voltage) * (1.0/3.57))
                current += self.weight * self.nmda_ratio * P_release * mg_block
                
        else:  # Inhibitory
            # GABAA current (fast)
            current -= self.weight * self.gabaa_ratio * P_release
            
            # GABAB current (slow)
            if self.gabab_ratio > 0:
                current -= self.weight * self.gabab_ratio * P_release * 0.5
        
        return current
    
    def update_structural_plasticity(self, dt: float):
        """Update structural plasticity based on activity and stability."""
        # Update stability based on weight
        if self.weight < self.pruning_threshold:
            self.stability -= dt * 0.1
        else:
            self.stability += dt * self.growth_factor
        self.stability = np.clip(self.stability, 0.0, 1.0)
        
        # Adjust weight based on stability
        if self.stability > 0.8:  # Stable synapse
            self.weight += dt * self.growth_factor
        elif self.stability < 0.2:  # Unstable synapse
            self.weight -= dt * 0.2
            
        self.weight = np.clip(self.weight, 
                            self.stdp_params.w_min, 
                            self.stdp_params.w_max)
    
    def record_activation(self, activation: float):
        """Record activation for metaplasticity."""
        self.activation_history.append(activation)
        if len(self.activation_history) > self.max_history_length:
            self.activation_history.pop(0)
    
    def get_average_activation(self, window: int = 100) -> float:
        """Get average activation over recent history."""
        if not self.activation_history:
            return 0.0
        window = min(window, len(self.activation_history))
        return np.mean(self.activation_history[-window:])
    
    def reset_traces(self):
        """Reset all plasticity traces."""
        self.pre_trace = 0.0
        self.post_trace = 0.0
        self.vesicle_pool = 1.0
        self.facilitation = 1.0