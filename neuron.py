# neuron.py
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict

class NeuronType(Enum):
    # Principal/Excitatory neurons
    PYRAMIDAL = "pyramidal"
    SPINY_STELLATE = "spiny_stellate"
    STAR_PYRAMID = "star_pyramid"
    
    # GABAergic interneurons
    PV = "parvalbumin"
    SST = "somatostatin"
    VIP = "vip"
    BASKET = "basket"
    MARTINOTTI = "martinotti"
    CHANDELIER = "chandelier"
    BIPOLAR = "bipolar"
    NEUROGLIAFORM = "neurogliaform"

class IonChannelType(Enum):
    NA = "sodium"        # Fast sodium channels
    K = "potassium"      # Delayed rectifier potassium
    KA = "potassium_a"   # A-type potassium
    KM = "potassium_m"   # M-type potassium
    CAL = "calcium_l"    # L-type calcium
    CAT = "calcium_t"    # T-type calcium
    H = "h_current"      # H-current (HCN)
    LEAK = "leak"        # Leak channels

class Synapse:
    """Implements synaptic dynamics including short-term plasticity."""
    def __init__(self, weight: float, E_rev: float, tau_rise: float, tau_decay: float):
        """
        Initialize a synapse.
        
        Args:
            weight: Synaptic weight (conductance scaling factor)
            E_rev: Reversal potential (mV)
            tau_rise: Rise time constant (ms)
            tau_decay: Decay time constant (ms)
        """
        self.weight = weight
        self.E_rev = E_rev
        self.tau_rise = tau_rise
        self.tau_decay = tau_decay
        
        # Synaptic conductance
        self.g = 0.0
        
        # Short-term plasticity variables
        self.depression = 1.0  # Available vesicle pool
        self.facilitation = 1.0  # Release probability scaling
        self.tau_depression = 800.0  # Recovery time constant (ms)
        self.tau_facilitation = 100.0  # Facilitation decay time constant (ms)
        
        # Last update time
        self.last_spike_time = -1000.0
        
    def update(self, t: float, dt: float, spike: bool):
        """
        Update synaptic state.
        
        Args:
            t: Current time (ms)
            dt: Time step (ms)
            spike: Whether there is a presynaptic spike
        """
        # Update short-term plasticity
        time_since_last = t - self.last_spike_time
        
        # Recovery of depression
        self.depression += dt * (1.0 - self.depression) / self.tau_depression
        
        # Decay of facilitation
        self.facilitation += dt * (1.0 - self.facilitation) / self.tau_facilitation
        
        if spike:
            # Calculate release probability
            P_release = self.facilitation * self.depression
            
            # Update depression and facilitation
            self.depression *= (1.0 - P_release)
            self.facilitation += 0.2  # Facilitation increment
            
            # Add new conductance
            self.g += self.weight * P_release
            
            self.last_spike_time = t
        
        # Decay existing conductance
        self.g *= np.exp(-dt / self.tau_decay)

@dataclass
class IonChannel:
    """Represents voltage-gated ion channels."""
    type: IonChannelType
    g_max: float  # Maximum conductance
    E_rev: float  # Reversal potential
    m: float = 0.0  # Activation variable
    h: float = 1.0  # Inactivation variable
    
    def alpha_m(self, v: float) -> float:
        """Calculate activation rate constant."""
        if self.type == IonChannelType.NA:
            return 0.1 * (v + 40) / (1 - np.exp(-(v + 40) / 10))
        elif self.type == IonChannelType.K:
            return 0.01 * (v + 55) / (1 - np.exp(-(v + 55) / 10))
        elif self.type == IonChannelType.CAL:
            return 0.055 * (v + 27) / (1 - np.exp(-(v + 27) / 3.8))
        return 0.0
    
    def beta_m(self, v: float) -> float:
        """Calculate deactivation rate constant."""
        if self.type == IonChannelType.NA:
            return 4.0 * np.exp(-(v + 65) / 18)
        elif self.type == IonChannelType.K:
            return 0.125 * np.exp(-(v + 65) / 80)
        elif self.type == IonChannelType.CAL:
            return 0.94 * np.exp(-(v + 75) / 17)
        return 0.0

class AdvancedNeuron:
    """Advanced neuron model with type-specific properties and detailed ion channels."""
    
    def __init__(self, idx: int, neuron_type: NeuronType):
        self.idx = idx
        self.type = neuron_type
        
        # Set type-specific parameters
        self._initialize_parameters()
        
        # Basic state variables
        self.v = self.E_L  # Membrane potential
        self.spike = False
        self.last_spike_time = -1000.0
        
        # Ion channels - initialized based on neuron type
        self.channels = self._initialize_channels()
        
        # Calcium dynamics
        self.Ca = 0.0
        self.Ca_buffer = 0.0
        self.tau_Ca = 50.0
        
        # Adaptation
        self.w = 0.0  # Adaptation current
        self.spike_threshold = self.V_th
        
        # Synaptic state
        self.synapses = {}  # source_idx -> Synapse
        self.dendritic_segments = []
        
        # Metabolic state
        self.atp = 1.0
        self.energy_usage = 0.0
        
    def _initialize_parameters(self):
        """Initialize parameters based on neuron type."""
        if self.type in [NeuronType.PYRAMIDAL, NeuronType.SPINY_STELLATE]:
            # Regular spiking excitatory neuron parameters
            self.C = 200.0      # Membrane capacitance (pF)
            self.g_L = 10.0     # Leak conductance (nS)
            self.E_L = -70.0    # Leak reversal potential (mV)
            self.V_th = -50.0   # Spike threshold (mV)
            self.delta_T = 2.0  # Slope factor (mV)
            self.tau_w = 300.0  # Adaptation time constant (ms)
            self.a = 2.0        # Subthreshold adaptation (nS)
            self.b = 60.0       # Spike-triggered adaptation (pA)
            self.V_reset = -60.0  # Reset potential (mV)
            self.t_ref = 2.0    # Refractory period (ms)
            
        elif self.type in [NeuronType.PV, NeuronType.BASKET]:
            # Fast-spiking interneuron parameters
            self.C = 100.0
            self.g_L = 10.0
            self.E_L = -65.0
            self.V_th = -45.0
            self.delta_T = 0.5
            self.tau_w = 30.0
            self.a = 0.0
            self.b = 0.0
            self.V_reset = -55.0
            self.t_ref = 0.5
            
        elif self.type in [NeuronType.SST, NeuronType.MARTINOTTI]:
            # Low-threshold spiking interneuron parameters
            self.C = 150.0
            self.g_L = 12.0
            self.E_L = -65.0
            self.V_th = -47.0
            self.delta_T = 1.0
            self.tau_w = 150.0
            self.a = 1.0
            self.b = 10.0
            self.V_reset = -60.0
            self.t_ref = 1.0
            
        else:  # Default parameters for other types
            self.C = 200.0
            self.g_L = 10.0
            self.E_L = -70.0
            self.V_th = -50.0
            self.delta_T = 2.0
            self.tau_w = 300.0
            self.a = 2.0
            self.b = 60.0
            self.V_reset = -60.0
            self.t_ref = 2.0

    def _initialize_channels(self) -> Dict[IonChannelType, IonChannel]:
        """Initialize ion channels based on neuron type."""
        channels = {
            IonChannelType.LEAK: IonChannel(IonChannelType.LEAK, self.g_L, self.E_L)
        }
        
        if self.type in [NeuronType.PYRAMIDAL, NeuronType.SPINY_STELLATE]:
            channels.update({
                IonChannelType.NA: IonChannel(IonChannelType.NA, 120.0, 50.0),
                IonChannelType.K: IonChannel(IonChannelType.K, 36.0, -77.0),
                IonChannelType.CAL: IonChannel(IonChannelType.CAL, 5.0, 120.0),
                IonChannelType.KM: IonChannel(IonChannelType.KM, 1.0, -80.0)
            })
            
        elif self.type in [NeuronType.PV, NeuronType.BASKET]:
            channels.update({
                IonChannelType.NA: IonChannel(IonChannelType.NA, 150.0, 50.0),
                IonChannelType.K: IonChannel(IonChannelType.K, 50.0, -77.0),
                IonChannelType.KA: IonChannel(IonChannelType.KA, 20.0, -80.0)
            })
            
        return channels

    def integrate(self, dt: float):
        """Integrate all neuronal dynamics."""
        # Update ion channels
        I_total = 0
        for channel in self.channels.values():
            channel.update(self.v, dt)
            I_total += channel.current(self.v)
        
        # Calcium dynamics
        cal_current = self.channels.get(IonChannelType.CAL, None)
        if cal_current:
            self.Ca += (
                -self.Ca / self.tau_Ca +  # Decay
                0.1 * cal_current.current(self.v)  # Calcium entry
            ) * dt
        
        # Adaptation current
        self.w += (
            self.a * (self.v - self.E_L) - self.w
        ) / self.tau_w * dt
        
        # Membrane potential update
        dv = (
            -I_total - self.w +  # Intrinsic currents
            self._synaptic_current()  # Synaptic input
        ) / self.C
        
        self.v += dv * dt
        
        # Spike generation
        if self.v >= self.spike_threshold:
            self.spike = True
            self.v = self.V_reset
            self.w += self.b  # Spike-triggered adaptation
            self.spike_threshold += 2.0  # Threshold adaptation
        else:
            self.spike = False
            self.spike_threshold += (self.V_th - self.spike_threshold) / 50.0 * dt
        
        # Update metabolic state
        self._update_metabolism(dt, abs(I_total))

    def _synaptic_current(self) -> float:
        """Calculate total synaptic current."""
        I_syn = 0.0
        for synapse in self.synapses.values():
            I_syn += synapse.g * (self.v - synapse.E_rev)
        return I_syn
    
    def _update_metabolism(self, dt: float, current_magnitude: float):
        """Update metabolic state based on activity."""
        # ATP consumption
        self.energy_usage = (
            0.01 * current_magnitude +  # Basic ion pump usage
            5.0 * float(self.spike)     # Action potential cost
        )
        self.atp -= self.energy_usage * dt
        
        # ATP recovery
        self.atp += (1.0 - self.atp) / 1000.0 * dt  # Slow recovery
        self.atp = np.clip(self.atp, 0.0, 1.0)

    def receive_input(self, source_idx: int, weight: float, spike_time: float):
        """Process incoming synaptic input."""
        if source_idx not in self.synapses:
            E_rev = 0.0 if self.type in [NeuronType.PYRAMIDAL, NeuronType.SPINY_STELLATE] else -80.0
            self.synapses[source_idx] = Synapse(weight, E_rev, 0.5, 5.0)
        
        self.synapses[source_idx].update(spike_time, 0.1, True)