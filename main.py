# main.py
import numpy as np
from visualization import RealtimeVisualizer
from cortical_layers import CorticalLayer
from checkpointing import ResumableSimulation
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
import subprocess
from auditory import AudioProcessor, RealtimeAudioSimulation
from sensory_input import MultimodalProcessor, MultimodalSimulation
from brain_cache import get_or_create_brain

def verify_dependencies():
    """Verify that all required dependencies are installed."""
    try:
        import cv2
        import librosa
        import soundfile
    except ImportError as e:
        print(f"Missing required dependency: {str(e)}")
        print("Please install required packages using:")
        print("pip install opencv-python librosa soundfile")
        sys.exit(1)
        
    # Check for ffmpeg installation
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
    except FileNotFoundError:
        print("ffmpeg is not installed. Please install ffmpeg:")
        print("On Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("On macOS with Homebrew: brew install ffmpeg")
        print("On Windows: Download from https://ffmpeg.org/download.html")
        sys.exit(1)

def process_multimodal(video_file: str, scale_factor: float = 0.001, force_new_brain: bool = False):
    """Process video file with synchronized audio and visual stimulation."""
    print("\n=== Neural Response to Audio-Visual Processing ===\n")
    
    # Initialize components
    print("1. Initializing brain and multimodal processor...")
    brain = get_or_create_brain(scale_factor=scale_factor, force_new=force_new_brain)
    multimodal_processor = MultimodalProcessor()
    simulation = MultimodalSimulation(brain, multimodal_processor)
    
    # Set up visualization
    print("2. Setting up visualization...")
    visualizer = RealtimeVisualizer()
    visualizer.setup()
    
    def visualization_callback(data):
        visualizer.update_data(data)
    
    simulation.register_callback(visualization_callback)
    
    try:
        print(f"3. Processing video file: {video_file}")
        import threading
        processing_thread = threading.Thread(
            target=simulation.process_video_file,
            args=(video_file,)
        )
        processing_thread.start()
        
        visualizer.start()
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        simulation.stop()
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        simulation.stop()
        raise
    finally:
        visualizer.stop()
        if processing_thread.is_alive():
            processing_thread.join(timeout=1.0)

def process_audio(audio_file: str, scale_factor: float = 0.001, force_new_brain: bool = False):
    """Process audio file and visualize brain activity."""
    print("\n=== Neural Response to Audio Processing ===\n")
    
    # Initialize components
    print("1. Initializing brain and audio processor...")
    brain = get_or_create_brain(scale_factor=scale_factor, force_new=force_new_brain)
    audio_processor = AudioProcessor()
    simulation = RealtimeAudioSimulation(brain, audio_processor)
    
    # Set up visualization
    print("2. Setting up visualization...")
    visualizer = RealtimeVisualizer()
    visualizer.setup()
    
    def visualization_callback(data):
        visualizer.update_data(data)
    
    simulation.register_callback(visualization_callback)
    
    try:
        print(f"3. Processing audio file: {audio_file}")
        import threading
        processing_thread = threading.Thread(
            target=simulation.process_audio_file,
            args=(audio_file,)
        )
        processing_thread.start()
        
        visualizer.start()
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        simulation.stop()
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        simulation.stop()
        raise
    finally:
        visualizer.stop()
        if processing_thread.is_alive():
            processing_thread.join(timeout=1.0)

def test_digital_brain(scale_factor=0.001, duration=1000.0, dt=0.1, resume_id=None, force_new_brain=False):
    """Comprehensive test of the digital brain implementation."""
    print("\n=== Digital Brain Test Suite ===\n")
    
    # Initialize brain with caching
    print("1. Initializing brain...")
    brain = get_or_create_brain(scale_factor=scale_factor, force_new=force_new_brain)
    
    print("2. Setting up simulation...")
    sim = ResumableSimulation(
        brain=brain,
        duration=duration,
        dt=dt,
        checkpoint_frequency=1000
    )
    
    # Basic structure tests
    print("\n3. Testing brain structure:")
    print(f"- Total neurons: {len(brain.neurons):,}")
    print(f"- Total connections: {len(brain.connections):,}")
    
    # Test region connectivity
    print("\n4. Testing region connectivity:")
    for region_name, region in brain.regions.items():
        outgoing = 0
        for (src, _) in brain.connections.keys():
            if any(layer_config.start_idx <= src < layer_config.end_idx 
                  for layer_config in region.layer_configs.values()):
                outgoing += 1
        print(f"- {region_name}: {outgoing:,} outgoing connections")
    
    # Print layer information
    print("\n5. Layer configuration:")
    for region_name, region in brain.regions.items():
        print(f"\n{region_name} layers:")
        for layer, config in region.layer_configs.items():
            neuron_count = config.end_idx - config.start_idx
            print(f"- {layer.name}: {neuron_count} neurons ({config.start_idx}-{config.end_idx})")
    
    # Test stimulation
    try:
        v1_l4_neurons = brain.get_layer_neurons('v1', CorticalLayer.L4)
        if v1_l4_neurons:
            print(f"\n6. Stimulating {min(100, len(v1_l4_neurons))} neurons in V1 Layer 4")
            stim_neurons = v1_l4_neurons[:100]
            for neuron in stim_neurons:
                neuron.receive_input(-1, 2.0, 0)
        else:
            print("\n6. Warning: No neurons found in V1 Layer 4")
    except Exception as e:
        print(f"\n6. Error stimulating V1 Layer 4: {str(e)}")
    
    # Save simulation ID
    sim_id = None
    if sim.checkpoint_manager.last_checkpoint:
        sim_id = sim.checkpoint_manager._get_simulation_id(sim)
        id_file = Path("last_simulation_id.txt")
        id_file.write_text(sim_id)
        print(f"\nSimulation ID saved to {id_file}")
    
    try:
        print("\n7. Running simulation...")
        sim.run(resume_id=resume_id)
        
        print("\n8. Analyzing simulation results:")
        analyze_simulation_results(sim)
        
        return sim
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        if sim_id:
            print(f"Can resume using simulation ID: {sim_id}")
        return None
    except Exception as e:
        print(f"\nError during simulation: {str(e)}")
        if sim_id:
            print(f"Can resume using simulation ID: {sim_id}")
        raise

def analyze_simulation_results(sim):
    """Analyze and plot simulation results."""
    # Calculate active neurons
    active_neurons = np.where(np.sum(sim.spike_record, axis=0) > 0)[0]
    
    if len(active_neurons) > 0:
        plt.figure(figsize=(12, 6))
        spike_times = np.where(sim.spike_record[:, active_neurons])[0]
        spike_neurons = np.where(sim.spike_record[:, active_neurons])[1]
        plt.scatter(spike_times * sim.dt, spike_neurons, s=1, c='black')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron Index')
        plt.title('Spike Raster Plot')
        plt.show()
    else:
        print("- Warning: No active neurons detected")
    
    # Firing rate analysis
    firing_rates = np.sum(sim.spike_record, axis=0) / (sim.duration/1000.0)
    mean_rate = np.mean(firing_rates)
    print(f"- Mean firing rate: {mean_rate:.2f} Hz")
    
    # Plot rate distribution
    plt.figure(figsize=(10, 4))
    plt.hist(firing_rates[firing_rates > 0], bins=50)
    plt.xlabel('Firing Rate (Hz)')
    plt.ylabel('Count')
    plt.title('Distribution of Firing Rates')
    plt.show()
    
    # Network state analysis
    print(f"- Final network state: {sim.state.value}")
    if sim.oscillations:
        print("- Detected oscillations:")
        for osc in sim.oscillations:
            print(f"  * {osc.frequency:.1f} Hz (power: {osc.power:.2f})")
    
    # Layer-specific activity
    print("\n9. Layer-specific activity:")
    for region_name, region in sim.brain.regions.items():
        print(f"\n{region_name}:")
        for layer, config in region.layer_configs.items():
            layer_rates = firing_rates[config.start_idx:config.end_idx]
            if len(layer_rates) > 0:
                active_count = np.sum(layer_rates > 0)
                mean_layer_rate = np.mean(layer_rates)
                print(f"- {layer.name}: {mean_layer_rate:.2f} Hz "
                      f"({active_count} active neurons)")
    
    # Connection analysis
    print("\n10. Connection analysis:")
    weights = [conn.weight for conn in sim.brain.connections.values()]
    print(f"- Mean weight: {np.mean(weights):.3f}")
    print(f"- Weight std: {np.std(weights):.3f}")
    print(f"- Min weight: {np.min(weights):.3f}")
    print(f"- Max weight: {np.max(weights):.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Digital Brain Simulation and Processing')
    parser.add_argument('--mode', choices=['test', 'audio', 'video', 'multimodal'], 
                      default='test',
                      help='Operation mode: test, audio, video, or multimodal processing')
    parser.add_argument('--input-file', type=str, 
                      help='Path to input file (audio or video)')
    parser.add_argument('--scale', type=float, default=0.001,
                      help='Brain scale factor (smaller = faster processing)')
    parser.add_argument('--resume-id', type=str, 
                      help='Resume from checkpoint ID')
    parser.add_argument('--force-new-brain', action='store_true',
                      help='Force creation of new brain instead of using cache')
    
    args = parser.parse_args()
    
    try:
        if args.mode in ['video', 'multimodal']:
            verify_dependencies()
        
        if args.mode in ['multimodal', 'video']:
            if not args.input_file:
                print("Error: --input-file required for video/multimodal processing")
                sys.exit(1)
            process_multimodal(args.input_file, args.scale, args.force_new_brain)
            
        elif args.mode == 'audio':
            if not args.input_file:
                print("Error: --input-file required for audio processing")
                sys.exit(1)
            process_audio(args.input_file, args.scale, args.force_new_brain)
            
        else:
            simulation = test_digital_brain(
                scale_factor=args.scale,
                duration=1000.0,
                dt=0.1,
                resume_id=args.resume_id,
                force_new_brain=args.force_new_brain
            )
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise