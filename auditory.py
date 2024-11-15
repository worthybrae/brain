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
    
    def map_features_to_stimulation(self, features: AudioFeatures, brain, timestep: int) -> Dict[str, np.ndarray]:
        """Map audio features to neural stimulation patterns."""
        stimulation_patterns = {}
        
        # Map different features to different brain regions and layers
        
        # V1 (Primary Visual Cortex) - Map tempo and onset information
        v1_l4_neurons = brain.get_layer_neurons('v1', CorticalLayer.L4)
        if v1_l4_neurons:
            # Create stimulation pattern based on onset strength
            v1_stim = np.zeros(len(v1_l4_neurons))
            onset_idx = min(timestep, len(features.onset_strength) - 1)
            v1_stim[:] = features.onset_strength[onset_idx] * 2.0  # Scale factor
            stimulation_patterns['v1_l4'] = v1_stim
            
        # V2 - Map spectral centroid
        v2_l4_neurons = brain.get_layer_neurons('v2', CorticalLayer.L4)
        if v2_l4_neurons:
            v2_stim = np.zeros(len(v2_l4_neurons))
            spec_idx = min(timestep, len(features.spectral_centroid) - 1)
            v2_stim[:] = features.spectral_centroid[spec_idx] / 1000.0  # Normalize
            stimulation_patterns['v2_l4'] = v2_stim
            
        # IT (Inferior Temporal) - Map MFCC features
        it_l4_neurons = brain.get_layer_neurons('it', CorticalLayer.L4)
        if it_l4_neurons:
            it_stim = np.zeros(len(it_l4_neurons))
            mfcc_idx = min(timestep, features.mfcc.shape[1] - 1)
            it_stim[:] = np.repeat(features.mfcc[:, mfcc_idx], 
                                 len(it_l4_neurons) // 13 + 1)[:len(it_l4_neurons)]
            stimulation_patterns['it_l4'] = it_stim
            
        return stimulation_patterns

class RealtimeAudioSimulation:
    """Manages real-time audio processing and brain simulation."""
    
    def __init__(self, brain: ModularDigitalBrain, audio_processor: AudioProcessor):
        self.brain = brain
        self.audio_processor = audio_processor
        self.simulation = ResumableSimulation(brain, duration=1000.0, dt=0.1)
        self.is_running = False
        self.current_time = 0
        self.callbacks = []
        
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
        
    def _run_simulation(self, features: AudioFeatures, duration: float):
        """Run simulation with audio-driven neural stimulation."""
        timestep = 0
        start_time = time.time()
        
        while self.is_running and self.current_time < duration:
            # Map audio features to neural stimulation
            stimulation = self.audio_processor.map_features_to_stimulation(
                features, self.brain, timestep
            )
            
            # Apply stimulation to neurons
            for region_name, stim_pattern in stimulation.items():
                if region_name == 'v1_l4':
                    neurons = self.brain.get_layer_neurons('v1', CorticalLayer.L4)
                    for i, neuron in enumerate(neurons):
                        neuron.receive_input(-1, stim_pattern[i], self.current_time)
            
            # Run one simulation timestep
            self.simulation._update_timestep(timestep)
            
            # Calculate network statistics
            stats = self._calculate_statistics(timestep)
            
            # Notify callbacks with new data
            self.notify_callbacks({
                'time': self.current_time,
                'stats': stats,
                'stimulation': stimulation
            })
            
            # Update timing
            timestep += 1
            self.current_time = time.time() - start_time
            
            # Maintain real-time simulation
            expected_time = timestep * self.simulation.dt / 1000.0
            if self.current_time < expected_time:
                time.sleep(expected_time - self.current_time)
                
    def _calculate_statistics(self, timestep: int) -> Dict:
        """Calculate relevant network statistics."""
        return {
            'mean_firing_rate': np.mean(self.simulation.firing_rates),
            'synchrony_index': self.simulation.synchrony_index,
            'network_state': self.simulation.state.value,
            'active_neurons': np.sum(self.simulation.spike_record[timestep] > 0),
            'region_activity': {
                region_name: np.mean([
                    n.v for layer_config in region.layer_configs.values()
                    for n in layer_config.neurons
                ]) if any(layer_config.neurons for layer_config in region.layer_configs.values()) else 0.0
                for region_name, region in self.brain.regions.items()
            }
        }
        
    def stop(self):
        """Stop the simulation."""
        self.is_running = False