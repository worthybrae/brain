import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from digital_brain import ModularDigitalBrain
from cortical_layers import CorticalLayer
from checkpointing import ResumableSimulation
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class VideoFeatures:
    """Features extracted from video frames."""
    edges: np.ndarray           # Edge detection map
    motion: np.ndarray         # Motion detection between frames
    colors: np.ndarray         # Color analysis
    orientations: np.ndarray   # Gabor filter responses
    textures: np.ndarray       # Texture features
    objects: List[Dict]        # Detected object regions

class VideoProcessor:
    """Processes video frames and converts features to neural stimulation patterns."""
    
    def __init__(self, frame_rate: int = 30):
        self.frame_rate = frame_rate
        self.prev_frame = None
        self.gabor_filters = self._create_gabor_filters()
        
        # Stimulation parameters
        self.visual_scale = 5.0
        self.motion_scale = 3.0
        self.edge_scale = 4.0
        self.color_scale = 2.0
        self.dt = 0.1
        
    def _create_gabor_filters(self):
        """Create Gabor filters for different orientations."""
        filters = []
        ksize = 31
        sigma = 4.0
        lambd = 10.0
        gamma = 0.5
        psi = 0
        
        for theta in np.arange(0, np.pi, np.pi / 8):
            kernel = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F
            )
            filters.append(kernel)
        return filters
    
    def extract_features(self, frame: np.ndarray) -> VideoFeatures:
        """Extract relevant features from video frame."""
        if frame is None:
            raise ValueError("Invalid frame provided")
            
        # Ensure frame is in RGB format
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            
        # Convert to grayscale for edge and motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Edge detection using Canny
        edges = cv2.Canny(gray, 100, 200)
        
        # Motion detection
        motion = np.zeros_like(gray, dtype=np.float32)
        if self.prev_frame is not None:
            prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_RGB2GRAY) if len(self.prev_frame.shape) == 3 else self.prev_frame
            motion = cv2.absdiff(gray, prev_gray)
        self.prev_frame = frame.copy()
        
        # Color analysis (mean color values in regions)
        colors = cv2.mean(frame)[:3]
        
        # Orientation detection using Gabor filters
        orientations = []
        for kernel in self.gabor_filters:
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            orientations.append(np.mean(filtered))
        orientations = np.array(orientations)
        
        # Basic texture analysis
        textures = cv2.resize(gray, (32, 32))
        
        # Object detection using contours
        objects = []
        try:
            # Threshold the grayscale image
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    objects.append({
                        'bbox': (x, y, w, h),
                        'area': area
                    })
        except Exception as e:
            print(f"Warning: Error in contour detection: {str(e)}")
            objects = []
        
        return VideoFeatures(
            edges=edges,
            motion=motion,
            colors=np.array(colors),
            orientations=orientations,
            textures=textures,
            objects=objects
        )
    
    def create_stim_pattern(self, neurons, base_signal, noise_level=0.2, baseline=0.1):
        """Create stimulation pattern with temporal and spatial variability."""
        if not neurons or len(neurons) == 0:
            return np.array([])
            
        n_neurons = len(neurons)
        temporal_noise = np.random.normal(0, noise_level, n_neurons)
        spatial_weights = np.random.lognormal(0, 0.5, n_neurons)
        spatial_weights = (spatial_weights - np.min(spatial_weights)) / (
            np.max(spatial_weights) - np.min(spatial_weights) + 1e-10
        )
        
        return baseline + (base_signal + temporal_noise) * spatial_weights
    
    def map_features_to_stimulation(self, features: VideoFeatures, brain, timestep: int) -> Dict[str, np.ndarray]:
        """Map video features to appropriate neural regions."""
        stimulation_patterns = {}
        
        # V1 - Basic feature processing
        v1_l4_neurons = brain.get_layer_neurons('v1', CorticalLayer.L4)
        if v1_l4_neurons:
            edge_strength = np.mean(features.edges) / 255.0
            orientation_strength = np.mean(np.abs(features.orientations))
            base_signal = (edge_strength * self.edge_scale + 
                         orientation_strength * self.visual_scale)
            v1_stim = self.create_stim_pattern(v1_l4_neurons, base_signal)
            if len(v1_stim) > 0:
                stimulation_patterns['v1_l4'] = v1_stim
        
        # V2 - Texture and motion
        v2_l4_neurons = brain.get_layer_neurons('v2', CorticalLayer.L4)
        if v2_l4_neurons:
            texture_activity = np.std(features.textures) / 255.0
            motion_strength = np.mean(features.motion) / 255.0
            base_signal = (texture_activity * self.visual_scale + 
                         motion_strength * self.motion_scale)
            v2_stim = self.create_stim_pattern(v2_l4_neurons, base_signal)
            if len(v2_stim) > 0:
                stimulation_patterns['v2_l4'] = v2_stim
        
        # V4 - Color processing
        v4_l4_neurons = brain.get_layer_neurons('v4', CorticalLayer.L4)
        if v4_l4_neurons:
            color_variation = np.std(features.colors)
            color_intensity = np.mean(features.colors)
            base_signal = (color_variation * self.color_scale + 
                         color_intensity * self.visual_scale / 255.0)
            v4_stim = self.create_stim_pattern(v4_l4_neurons, base_signal)
            if len(v4_stim) > 0:
                stimulation_patterns['v4_l4'] = v4_stim
        
        # IT - Object processing
        it_l4_neurons = brain.get_layer_neurons('it', CorticalLayer.L4)
        if it_l4_neurons:
            object_activity = len(features.objects) * 0.1
            if features.objects:
                max_area = max(obj['area'] for obj in features.objects)
                object_activity += max_area / (features.edges.shape[0] * features.edges.shape[1])
            base_signal = object_activity * self.visual_scale
            it_stim = self.create_stim_pattern(it_l4_neurons, base_signal)
            if len(it_stim) > 0:
                stimulation_patterns['it_l4'] = it_stim
                
        return stimulation_patterns

class RealtimeVideoSimulation:
    """Manages real-time video processing and brain simulation."""
    
    def __init__(self, brain: ModularDigitalBrain, video_processor: VideoProcessor):
        self.brain = brain
        self.video_processor = video_processor
        self.simulation = ResumableSimulation(brain, duration=1000.0, dt=0.1)
        self.is_running = False
        self.current_time = 0
        self.callbacks = []
        
        # Initialize spike buffer
        self.spike_buffer_size = 1000
        self.spike_buffer = np.zeros((self.spike_buffer_size, len(brain.neurons)), dtype=bool)
        self.buffer_index = 0
        
        # Make neurons more responsive
        for neuron in brain.neurons:
            neuron.V_th -= 10.0
            neuron.g_L *= 1.5
            neuron.b *= 0.5
            neuron.t_ref = 1.0
            for syn in neuron.synapses.values():
                syn.weight *= 2.0
    
    def process_video_file(self, file_path: str):
        """Process video file and start simulation."""
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {file_path}")
        
        self.is_running = True
        frame_time = 1.0 / self.video_processor.frame_rate
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract features
                features = self.video_processor.extract_features(frame)
                
                # Map features to stimulation
                stimulation = self.video_processor.map_features_to_stimulation(
                    features, self.brain, int(self.current_time / self.simulation.dt)
                )
                
                # Update simulation
                self._update_simulation(stimulation)
                
                # Control timing
                time.sleep(frame_time)
                
        finally:
            cap.release()
            self.is_running = False
    
    def _update_simulation(self, stimulation):
        """Update simulation with current stimulation patterns."""
        timestep = int(self.current_time / self.simulation.dt)
        
        # Track spikes this timestep
        current_spikes = np.zeros(len(self.brain.neurons), dtype=bool)
        
        # Apply stimulation
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
        
        # Update spike buffer
        self.spike_buffer[self.buffer_index] = current_spikes
        self.buffer_index = (self.buffer_index + 1) % self.spike_buffer_size
        
        # Run simulation timestep
        self.simulation._update_timestep(timestep)
        
        # Calculate statistics
        stats = self._calculate_statistics(timestep, current_spikes)
        
        # Notify callbacks
        self._notify_callbacks({
            'time': self.current_time,
            'stats': stats,
            'spike_record': self.get_recent_spikes(),
            'stimulation': stimulation
        })
        
        self.current_time += self.simulation.dt
    
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