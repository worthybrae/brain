import numpy as np
import librosa
from typing import Dict, Tuple
from dataclasses import dataclass
import threading
import queue
import time
from cortical_layers import CorticalLayer
from digital_brain import ModularDigitalBrain
from checkpointing import ResumableSimulation

@dataclass
class AudioFeatures:
    """Features extracted from audio signal."""
    tempo: float
    spectral_centroid: np.ndarray
    mfcc: np.ndarray
    onset_strength: np.ndarray
    chromagram: np.ndarray
    
class AudioProcessor:
    """Processes audio files and converts features to neural stimulation patterns."""
    
    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.feature_queue = queue.Queue()
        self.is_processing = False
        
        # Stimulation parameters
        self.audio_scale = 5.0      # Base stimulation strength for audio
        self.visual_scale = 3.0     # Base stimulation strength for visual (future use)
        self.onset_scale = 3.0      # Scale factor for onset detection
        self.spectral_scale = 2.0   # Scale factor for spectral features
        self.dt = 0.1                # Default timestep in ms
        
    def _safe_normalize(self, weights):
        """Safely normalize weights array, handling edge cases."""
        if len(weights) == 0:
            return np.array([])
        if len(weights) == 1:
            return np.array([1.0])
        if np.all(weights == weights[0]):
            return np.ones_like(weights)
        
        min_val = np.min(weights)
        max_val = np.max(weights)
        if max_val == min_val:
            return np.ones_like(weights)
        return (weights - min_val) / (max_val - min_val)
    
    def _ensure_scalar(self, value):
        """Convert arrays or lists to scalar values safely."""
        if isinstance(value, (np.ndarray, list)):
            if len(value) > 0:
                return float(np.mean(value))
            return 0.0
        return float(value)
        
    def create_stim_pattern(self, neurons, base_signal, noise_level=0.2, baseline=0.1):
        """Create stimulation pattern with temporal and spatial variability."""
        if not neurons or len(neurons) == 0:
            return np.array([])
            
        # Ensure base_signal is a scalar
        base_signal = self._ensure_scalar(base_signal)
            
        n_neurons = len(neurons)
        temporal_noise = np.random.normal(0, noise_level, n_neurons)
        spatial_weights = np.random.lognormal(0, 0.5, n_neurons)
        
        # Safely normalize spatial weights
        spatial_weights = self._safe_normalize(spatial_weights)
        
        return baseline + (base_signal + temporal_noise) * spatial_weights
        
    def map_features_to_stimulation(self, features: AudioFeatures, brain, timestep: int) -> Dict[str, np.ndarray]:
        """Map audio features to appropriate neural regions based on processing mode."""
        stimulation_patterns = {}
        
        # Get current onset value for all regions
        onset_idx = min(timestep, len(features.onset_strength) - 1)
        onset_value = float(features.onset_strength[onset_idx])
        
        # 1. Primary Auditory Cortex (A1) - Processes basic sound features
        a1_l4_neurons = brain.get_layer_neurons('a1', CorticalLayer.L4)
        if a1_l4_neurons and len(a1_l4_neurons) > 0:
            # Map frequency information (spectral centroid and MFCC)
            spec_idx = min(timestep, len(features.spectral_centroid) - 1)
            spec_value = float(features.spectral_centroid[spec_idx])
            
            # Combine spectral and onset information
            base_signal = (spec_value * self.spectral_scale + 
                         onset_value * self.onset_scale) * self.audio_scale
            
            a1_stim = self.create_stim_pattern(a1_l4_neurons, base_signal, noise_level=0.15)
            if len(a1_stim) > 0:
                stimulation_patterns['a1_l4'] = a1_stim
        
        # 2. Secondary Auditory Cortex (A2) - Processes more complex sound features
        a2_l4_neurons = brain.get_layer_neurons('a2', CorticalLayer.L4)
        if a2_l4_neurons and len(a2_l4_neurons) > 0:
            # Map temporal features (rhythm, tempo)
            tempo_factor = np.sin(2 * np.pi * features.tempo * timestep * self.dt / 1000)
            
            # Combine with onset strength for rhythm processing
            base_signal = (0.5 + 0.5 * tempo_factor + 
                         0.3 * onset_value) * self.audio_scale
            
            a2_stim = self.create_stim_pattern(a2_l4_neurons, base_signal, noise_level=0.2)
            if len(a2_stim) > 0:
                stimulation_patterns['a2_l4'] = a2_stim
        
        # 3. Auditory Association Cortex (AA) - Processes complex patterns
        aa_l4_neurons = brain.get_layer_neurons('aa', CorticalLayer.L4)
        if aa_l4_neurons and len(aa_l4_neurons) > 0:
            # Process MFCC for timbre and phonetic content
            mfcc_idx = min(timestep, features.mfcc.shape[1] - 1)
            current_mfcc = features.mfcc[:, mfcc_idx]
            
            # Calculate MFCC change
            if timestep > 0:
                prev_mfcc = features.mfcc[:, max(0, mfcc_idx - 1)]
                mfcc_delta = np.mean(np.abs(current_mfcc - prev_mfcc))
            else:
                mfcc_delta = 0
                
            base_signal = (np.mean(current_mfcc) + 0.5 * mfcc_delta) * self.audio_scale
            
            aa_stim = self.create_stim_pattern(aa_l4_neurons, base_signal, noise_level=0.25)
            if len(aa_stim) > 0:
                stimulation_patterns['aa_l4'] = aa_stim
        
        # 4. Inferior Colliculus (IC) - Subcortical auditory processing
        ic_neurons = brain.get_layer_neurons('ic', CorticalLayer.L4)
        if ic_neurons and len(ic_neurons) > 0:
            # Process pitch information from chromagram
            if timestep < features.chromagram.shape[1]:
                chroma_current = features.chromagram[:, timestep]
                pitch_strength = np.max(chroma_current)
                pitch_complexity = -np.sum(chroma_current * np.log(chroma_current + 1e-10))
                
                base_signal = (pitch_strength + 0.3 * pitch_complexity) * self.audio_scale
                
                ic_stim = self.create_stim_pattern(ic_neurons, base_signal, noise_level=0.15)
                if len(ic_stim) > 0:
                    stimulation_patterns['ic_l4'] = ic_stim
        
        # 5. Superior Temporal Gyrus (STG) - High-level auditory processing
        stg_l4_neurons = brain.get_layer_neurons('stg', CorticalLayer.L4)
        if stg_l4_neurons and len(stg_l4_neurons) > 0:
            # Integrate multiple features for high-level processing
            if timestep < features.chromagram.shape[1]:
                # Combine rhythm, pitch, and spectral features
                high_level_features = [
                    float(onset_value),
                    float(0.5 + 0.5 * np.sin(2 * np.pi * features.tempo * timestep * self.dt / 1000)),
                    float(np.mean(features.chromagram[:, timestep])),
                    float(features.spectral_centroid[min(timestep, len(features.spectral_centroid) - 1)])
                ]
                
                base_signal = np.mean(high_level_features) * self.audio_scale
                
                stg_stim = self.create_stim_pattern(stg_l4_neurons, base_signal, noise_level=0.2)
                if len(stg_stim) > 0:
                    stimulation_patterns['stg_l4'] = stg_stim
        
        # 6. Prefrontal Cortex (PFC) - Only stimulate for significant events
        pfc_l4_neurons = brain.get_layer_neurons('pfc', CorticalLayer.L4)
        if pfc_l4_neurons and len(pfc_l4_neurons) > 0:
            # Detect significant audio events
            if onset_value > 0.8 or np.mean(features.spectral_centroid) > 0.9:
                base_signal = 0.5 * self.audio_scale
                pfc_stim = self.create_stim_pattern(pfc_l4_neurons, base_signal, noise_level=0.1)
                if len(pfc_stim) > 0:
                    stimulation_patterns['pfc_l4'] = pfc_stim
        
        return stimulation_patterns
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, float]:
        """Load and preprocess audio file."""
        # Load audio file
        audio, _ = librosa.load(file_path, sr=self.sample_rate)
        duration = librosa.get_duration(y=audio, sr=self.sample_rate)
        return audio, duration
        
    def extract_features(self, audio: np.ndarray) -> AudioFeatures:
        """Extract relevant features from audio signal."""
        # Tempo and beat information
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        
        # MFCCs (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        
        # Onset strength
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        
        # Chromagram
        chromagram = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        
        return AudioFeatures(
            tempo=tempo,
            spectral_centroid=spectral_centroid,
            mfcc=mfcc,
            onset_strength=onset_env,
            chromagram=chromagram
        )

class RealtimeAudioSimulation:
    """Manages real-time audio processing and brain simulation."""
    
    def __init__(self, brain: ModularDigitalBrain, audio_processor: AudioProcessor):
        self.brain = brain
        self.audio_processor = audio_processor
        self.simulation = ResumableSimulation(brain, duration=1000.0, dt=0.1)
        
        # Increase neural excitability even more aggressively
        for neuron in brain.neurons:
            # Make neurons much more responsive
            neuron.V_th -= 10.0    # Lower spike threshold more significantly
            neuron.g_L *= 1.5      # Increase membrane conductance further
            neuron.b *= 0.5        # Reduce adaptation more significantly
            neuron.t_ref = 1.0     # Reduce refractory period
            
            # Adjust synaptic weights to increase network connectivity
            for syn in neuron.synapses.values():
                syn.weight *= 2.0   # Double synaptic weights
        
        # Add spike history buffer
        self.spike_buffer_size = 1000  # 100ms at 0.1ms timestep
        self.spike_buffer = np.zeros((self.spike_buffer_size, len(brain.neurons)), dtype=bool)
        self.buffer_index = 0
        
        self.is_running = False
        self.current_time = 0
        self.callbacks = []
        
    def _run_simulation(self, features: AudioFeatures, duration: float):
        """Run simulation with audio-driven neural stimulation."""
        timestep = 0
        start_time = time.time()
        
        while self.is_running and self.current_time < duration:
            # Map audio features to neural stimulation
            stimulation = self.audio_processor.map_features_to_stimulation(
                features, self.brain, timestep
            )
            
            # Track spikes this timestep
            current_spikes = np.zeros(len(self.brain.neurons), dtype=bool)
            
            # Apply stimulation to neurons with increased strength
            for region_name, stim_pattern in stimulation.items():
                if region_name.endswith('_l4'):
                    region_name = region_name[:-3]
                    neurons = self.brain.get_layer_neurons(region_name, CorticalLayer.L4)
                    if neurons:
                        for i, neuron in enumerate(neurons):
                            # Increase stimulation strength significantly
                            stim_value = stim_pattern[i] * 5.0 + 0.5  # More aggressive scaling
                            neuron.receive_input(-1, stim_value, self.current_time)
                            
                            # Record spike if it occurred
                            if neuron.spike:
                                current_spikes[neuron.idx] = True
            
            # Update spike buffer
            self.spike_buffer[self.buffer_index] = current_spikes
            self.buffer_index = (self.buffer_index + 1) % self.spike_buffer_size
            
            # Run simulation timestep
            self.simulation._update_timestep(timestep)
            
            # Calculate network statistics
            stats = self._calculate_statistics(timestep, current_spikes)
            
            # Notify callbacks with new data
            self.notify_callbacks({
                'time': self.current_time,
                'stats': stats,
                'spike_record': self.get_recent_spikes(),
                'stimulation': stimulation
            })
            
            # Update timing
            timestep += 1
            self.current_time = time.time() - start_time
            
            # Maintain real-time simulation
            expected_time = timestep * self.simulation.dt / 1000.0
            if self.current_time < expected_time:
                time.sleep(expected_time - self.current_time)
    
    def get_recent_spikes(self) -> np.ndarray:
        """Get the most recent spike data from the circular buffer."""
        # Reconstruct proper time ordering from circular buffer
        ordered_spikes = np.zeros_like(self.spike_buffer)
        ordered_spikes[:self.spike_buffer_size - self.buffer_index] = \
            self.spike_buffer[self.buffer_index:]
        ordered_spikes[self.spike_buffer_size - self.buffer_index:] = \
            self.spike_buffer[:self.buffer_index]
        return ordered_spikes
    
    def _calculate_statistics(self, timestep: int, current_spikes: np.ndarray) -> Dict:
        """Calculate relevant network statistics using both buffer and simulation history."""
        # Calculate instantaneous firing rate
        instant_rate = np.mean(current_spikes) * 1000 / self.simulation.dt
        
        # Calculate short-term firing rate from recent buffer history
        recent_window = 1000  # 100ms at 0.1ms timestep
        start_idx = max(0, self.buffer_index - recent_window)
        if start_idx < self.buffer_index:
            recent_spikes = self.spike_buffer[start_idx:self.buffer_index]
        else:
            # Handle buffer wraparound
            recent_spikes = np.concatenate([
                self.spike_buffer[start_idx:],
                self.spike_buffer[:self.buffer_index]
            ])
        short_term_rate = np.mean(recent_spikes) * 1000 / (recent_window * self.simulation.dt)
        
        # Calculate longer-term firing rate from simulation history if available
        if hasattr(self.simulation, 'spike_record') and timestep > 0:
            history_window = min(10000, timestep)  # Use up to last 1 second
            long_term_rate = np.mean(self.simulation.spike_record[max(0, timestep - history_window):timestep]) * \
                            1000 / (history_window * self.simulation.dt)
        else:
            long_term_rate = short_term_rate
        
        # Calculate region activity using both membrane potential and recent spikes
        region_activity = {}
        region_details = {}
        for region_name, region in self.brain.regions.items():
            region_neurons = []
            region_indices = []
            for layer_config in region.layer_configs.values():
                if layer_config.neurons:
                    region_neurons.extend(layer_config.neurons)
                    region_indices.extend(range(layer_config.start_idx, layer_config.end_idx))
            
            if region_neurons:
                # Combine membrane potential and spike rate
                v_mean = np.mean([n.v for n in region_neurons])
                spike_rate = np.mean(current_spikes[region_indices]) * 1000 / self.simulation.dt
                region_details[region_name] = {
                    'voltage': v_mean,
                    'spike_rate': spike_rate,
                    'combined_activity': (v_mean + 70) / 100 + spike_rate
                }
                region_activity[region_name] = region_details[region_name]['combined_activity']
        
        # Calculate synchrony over different timescales
        instant_sync = np.sum(current_spikes) / len(self.brain.neurons)
        
        if len(recent_spikes) > 0:
            try:
                active_mask = np.any(recent_spikes, axis=0)
                if np.sum(active_mask) > 1:
                    active_spikes = recent_spikes[:, active_mask]
                    sync_index = np.corrcoef(active_spikes.T)
                    upper_triangle = sync_index[np.triu_indices_from(sync_index, k=1)]
                    sync_index = np.nanmean(np.abs(upper_triangle))
                else:
                    sync_index = 0.0
            except (ValueError, RuntimeWarning):
                sync_index = 0.0
        else:
            sync_index = 0.0
            
        if not np.isfinite(sync_index):
            sync_index = 0.0
        
        # Determine network state based on activity patterns
        if instant_rate > 50:
            network_state = "burst"
        elif sync_index > 0.3:
            network_state = "synchronized"
        elif short_term_rate > 5:
            network_state = "active"
        else:
            network_state = "resting"
        
        return {
            'timestep': timestep,
            'mean_firing_rate': short_term_rate,
            'instant_rate': instant_rate,
            'long_term_rate': long_term_rate,
            'network_state': network_state,
            'synchrony_index': sync_index,
            'instant_sync': instant_sync,
            'active_neurons': np.sum(current_spikes),
            'active_fraction': np.mean(current_spikes),
            'region_activity': region_activity,
            'region_details': region_details
        }
        
    def register_callback(self, callback):
        """Register callback for real-time updates."""
        self.callbacks.append(callback)
        
    def notify_callbacks(self, data):
        """Notify all registered callbacks with new data."""
        for callback in self.callbacks:
            callback(data)
            
    def process_audio_file(self, file_path: str):
        """Process audio file and start simulation."""
        # Load and process audio
        audio, duration = self.audio_processor.load_audio(file_path)
        features = self.audio_processor.extract_features(audio)
        
        # Start real-time simulation thread
        self.is_running = True
        simulation_thread = threading.Thread(
            target=self._run_simulation,
            args=(features, duration)
        )
        simulation_thread.start()
    
    def stop(self):
        """Stop the simulation."""
        self.is_running = False
            
    