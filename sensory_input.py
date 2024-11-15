import cv2
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
import threading
import queue
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from digital_brain import ModularDigitalBrain
from cortical_layers import CorticalLayer
from checkpointing import ResumableSimulation
from auditory import AudioProcessor, AudioFeatures
from visual import VideoProcessor, VideoFeatures

@dataclass
class MultimodalFeatures:
    """Combined audio and video features."""
    audio: Optional[AudioFeatures]
    video: Optional[VideoFeatures]
    timestamp: float

class MultimodalProcessor:
    """Processes combined audio and video streams."""
    
    def __init__(self, sample_rate: int = 22050, frame_rate: int = 30):
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        self.video_processor = VideoProcessor(frame_rate=frame_rate)
        
        # Processing queues
        self.audio_queue = queue.Queue()
        self.video_queue = queue.Queue()
        self.feature_queue = queue.Queue()
        
        # Synchronization
        self.is_processing = False
        self.audio_thread = None
        self.video_thread = None
        
    def extract_audio_from_video(self, video_path: str) -> Tuple[str, float]:
        """Extract audio from video file and save to temporary file."""
        # Create temporary file for audio
        temp_dir = tempfile.gettempdir()
        temp_audio_path = os.path.join(temp_dir, 'temp_audio.wav')
        
        # Use ffmpeg to extract audio
        import subprocess
        command = [
            'ffmpeg', '-i', video_path,
            '-ab', '160k', '-ac', '2', '-ar', str(self.sample_rate),
            '-vn', temp_audio_path, '-y'
        ]
        
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Get audio duration
        audio_info = sf.info(temp_audio_path)
        duration = audio_info.duration
        
        return temp_audio_path, duration
    
    def process_audio_stream(self, audio_path: str, duration: float):
        """Process audio stream in separate thread."""
        # Load audio
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        
        # Extract features
        features = self.audio_processor.extract_features(audio)
        
        # Calculate time per frame
        time_per_frame = 1.0 / self.frame_rate
        current_time = 0
        
        while self.is_processing and current_time < duration:
            # Find current audio frame
            frame_idx = int(current_time * self.sample_rate)
            
            # Put features in queue
            self.audio_queue.put((current_time, features))
            
            # Wait for next frame
            current_time += time_per_frame
            
    def process_video_stream(self, video_path: str):
        """Process video stream in separate thread."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        try:
            frame_count = 0
            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract features
                features = self.video_processor.extract_features(frame)
                
                # Calculate timestamp
                timestamp = frame_count / self.frame_rate
                
                # Put features in queue
                self.video_queue.put((timestamp, features))
                
                frame_count += 1
                
        finally:
            cap.release()
    
    def combine_features(self):
        """Combine audio and video features with synchronization."""
        audio_buffer = {}
        video_buffer = {}
        
        while self.is_processing:
            # Get latest audio and video features
            try:
                audio_time, audio_features = self.audio_queue.get(timeout=0.1)
                audio_buffer[audio_time] = audio_features
            except queue.Empty:
                pass
                
            try:
                video_time, video_features = self.video_queue.get(timeout=0.1)
                video_buffer[video_time] = video_features
            except queue.Empty:
                pass
            
            # Find matching timestamps
            audio_times = sorted(audio_buffer.keys())
            video_times = sorted(video_buffer.keys())
            
            if audio_times and video_times:
                # Find closest matching timestamps
                for v_time in video_times:
                    closest_a_time = min(audio_times, 
                                       key=lambda x: abs(x - v_time))
                    
                    # If timestamps are close enough, combine features
                    if abs(v_time - closest_a_time) < 1.0/self.frame_rate:
                        combined = MultimodalFeatures(
                            audio=audio_buffer[closest_a_time],
                            video=video_buffer[v_time],
                            timestamp=v_time
                        )
                        self.feature_queue.put(combined)
                        
                        # Clean up processed features
                        del audio_buffer[closest_a_time]
                        del video_buffer[v_time]
            
            # Clean up old features
            current_time = max(video_times) if video_times else 0
            audio_buffer = {t: f for t, f in audio_buffer.items() 
                          if current_time - t < 1.0}
            video_buffer = {t: f for t, f in video_buffer.items() 
                          if current_time - t < 1.0}

class MultimodalSimulation:
    """Manages real-time audio-visual processing and brain simulation."""
    
    def __init__(self, brain: ModularDigitalBrain, multimodal_processor: MultimodalProcessor):
        self.brain = brain
        self.processor = multimodal_processor
        self.simulation = ResumableSimulation(brain, duration=1000.0, dt=0.1)
        
        # Make neurons more responsive
        for neuron in brain.neurons:
            neuron.V_th -= 10.0
            neuron.g_L *= 1.5
            neuron.b *= 0.5
            neuron.t_ref = 1.0
            for syn in neuron.synapses.values():
                syn.weight *= 2.0
        
        # Initialize spike buffer
        self.spike_buffer_size = 1000
        self.spike_buffer = np.zeros((self.spike_buffer_size, len(brain.neurons)), dtype=bool)
        self.buffer_index = 0
        
        self.is_running = False
        self.current_time = 0
        self.callbacks = []
        
    def process_video_file(self, file_path: str):
        """Process video file with synchronized audio and visual stimulation."""
        # Extract audio
        audio_path, duration = self.processor.extract_audio_from_video(file_path)
        
        try:
            # Start processing
            self.is_running = True
            self.processor.is_processing = True
            
            # Start audio processing thread
            self.processor.audio_thread = threading.Thread(
                target=self.processor.process_audio_stream,
                args=(audio_path, duration)
            )
            self.processor.audio_thread.start()
            
            # Start video processing thread
            self.processor.video_thread = threading.Thread(
                target=self.processor.process_video_stream,
                args=(file_path,)
            )
            self.processor.video_thread.start()
            
            # Start feature combination thread
            combine_thread = threading.Thread(
                target=self.processor.combine_features
            )
            combine_thread.start()
            
            # Process combined features
            while self.is_running:
                try:
                    features = self.processor.feature_queue.get(timeout=0.1)
                    self._process_features(features)
                except queue.Empty:
                    continue
                    
        finally:
            # Cleanup
            self.is_running = False
            self.processor.is_processing = False
            
            if self.processor.audio_thread:
                self.processor.audio_thread.join()
            if self.processor.video_thread:
                self.processor.video_thread.join()
            if combine_thread:
                combine_thread.join()
                
            # Remove temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
    
    def _process_features(self, features: MultimodalFeatures):
        """Process combined features and update simulation."""
        timestep = int(self.current_time / self.simulation.dt)
        
        # Track spikes this timestep
        current_spikes = np.zeros(len(self.brain.neurons), dtype=bool)
        
        # Process audio features
        if features.audio:
            audio_stim = self.processor.audio_processor.map_features_to_stimulation(
                features.audio, self.brain, timestep
            )
            self._apply_stimulation(audio_stim, current_spikes)
        
        # Process video features
        if features.video:
            video_stim = self.processor.video_processor.map_features_to_stimulation(
                features.video, self.brain, timestep
            )
            self._apply_stimulation(video_stim, current_spikes)
        
        # Update spike buffer
        self.spike_buffer[self.buffer_index] = current_spikes
        self.buffer_index = (self.buffer_index + 1) % self.spike_buffer_size
        
        # Run simulation timestep
        self.simulation._update_timestep(timestep)
        
        # Calculate statistics
        stats = self._calculate_statistics(timestep, current_spikes)
        
        # Notify callbacks
        combined_stim = {}
        if features.audio:
            combined_stim.update(audio_stim)
        if features.video:
            combined_stim.update(video_stim)
            
        self._notify_callbacks({
            'time': self.current_time,
            'stats': stats,
            'spike_record': self.get_recent_spikes(),
            'stimulation': combined_stim
        })
        
        self.current_time += self.simulation.dt
    
    def _apply_stimulation(self, stimulation: Dict[str, np.ndarray], 
                          current_spikes: np.ndarray):
        """Apply stimulation patterns to neurons."""
        for region_name, stim_pattern in stimulation.items():
            if region_name.endswith('_l4'):
                region_name = region_name[:-3]
                neurons = self.brain.get_layer_neurons(region_name, CorticalLayer.L4)
                if neurons:
                    for i, neuron in enumerate(neurons):
                        stim_value = stim_pattern[i] * 5.0 + 0.5
                        neuron.receive_input(-1, stim_value, self.current_time)
                        if neuron.spike:
                            current_spikes[neuron.idx] = True
    
    def get_recent_spikes(self) -> np.ndarray:
        """Get recent spike history from buffer."""
        ordered_spikes = np.zeros_like(self.spike_buffer)
        ordered_spikes[:self.spike_buffer_size - self.buffer_index] = \
            self.spike_buffer[self.buffer_index:]
        ordered_spikes[self.spike_buffer_size - self.buffer_index:] = \
            self.spike_buffer[:self.buffer_index]
        return ordered_spikes
    
    def register_callback(self, callback):
        """Register callback for real-time updates."""
        self.callbacks.append(callback)
    
    def _notify_callbacks(self, data):
        """Notify all registered callbacks with new data."""
        for callback in self.callbacks:
            callback(data)
    
    def stop(self):
        """Stop the simulation."""
        self.is_running = False