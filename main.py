# main.py
import numpy as np
from digital_brain import ModularDigitalBrain
from cortical_layers import CorticalLayer
from checkpointing import ResumableSimulation
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
import time
from auditory import AudioProcessor, RealtimeAudioSimulation
from brain_cache import get_or_create_brain

def setup_visualization():
    """Set up real-time matplotlib visualization."""
    plt.ion()  # Enable interactive mode
    fig = plt.figure(figsize=(15, 8))
    
    # Create subplots
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)  # Firing rate plot
    ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)  # Raster plot
    ax3 = plt.subplot2grid((3, 2), (2, 0))  # Region activity
    ax4 = plt.subplot2grid((3, 2), (2, 1))  # Network state
    
    ax1.set_title('Mean Firing Rate')
    ax2.set_title('Spike Raster')
    ax3.set_title('Region Activity')
    ax4.set_title('Network State')
    
    plt.tight_layout()
    return fig, (ax1, ax2, ax3, ax4)

def update_plots(data, fig, axes, rate_history, time_history):
    """Update visualization with new data."""
    ax1, ax2, ax3, ax4 = axes
    
    # Update firing rate history
    rate_history.append(data['stats']['mean_firing_rate'])
    time_history.append(data['time'])
    
    # Limit history length
    max_history = 1000
    if len(rate_history) > max_history:
        rate_history.pop(0)
        time_history.pop(0)
    
    # Update firing rate plot
    ax1.clear()
    ax1.plot(time_history, rate_history)
    ax1.set_title('Mean Firing Rate')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Rate (Hz)')
    
    # Update raster plot (last 100ms of spikes)
    ax2.clear()
    if 'spike_record' in data:
        recent_spikes = data['spike_record'][-100:]
        spike_times, spike_neurons = np.where(recent_spikes)
        ax2.scatter(spike_times, spike_neurons, s=1, c='black')
    ax2.set_title('Spike Raster (last 100ms)')
    
    # Update region activity
    ax3.clear()
    regions = list(data['stats']['region_activity'].keys())
    activities = [data['stats']['region_activity'][r] for r in regions]
    ax3.bar(regions, activities)
    ax3.set_title('Region Activity')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # Update network state
    ax4.clear()
    ax4.text(0.5, 0.5, f"State: {data['stats']['network_state']}\n"
             f"Sync Index: {data['stats']['synchrony_index']:.2f}\n"
             f"Active Neurons: {data['stats']['active_neurons']}",
             horizontalalignment='center',
             verticalalignment='center')
    ax4.axis('off')
    
    fig.canvas.draw()
    fig.canvas.flush_events()

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
    fig, axes = setup_visualization()
    rate_history = []
    time_history = []
    
    # Define visualization callback
    def visualization_callback(data):
        update_plots(data, fig, axes, rate_history, time_history)
    
    # Register callback
    simulation.register_callback(visualization_callback)
    
    try:
        print(f"3. Processing audio file: {audio_file}")
        simulation.process_audio_file(audio_file)
        
        # Keep main thread alive while processing
        while simulation.is_running:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        simulation.stop()
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        simulation.stop()
        raise
    finally:
        plt.ioff()
        plt.close(fig)

def test_digital_brain(scale_factor=0.001, duration=1000.0, dt=0.1, resume_id=None, force_new_brain=False):
    """
    Comprehensive test of the digital brain implementation with checkpoint support.
    
    Args:
        scale_factor: Brain size scaling factor (smaller = faster testing)
        duration: Simulation duration in ms
        dt: Integration time step in ms
        resume_id: Optional ID to resume from previous checkpoint
        force_new_brain: If True, force creation of new brain instead of using cache
    """
    print("\n=== Digital Brain Test Suite ===\n")
    
    # Initialize brain with caching
    print("1. Initializing brain...")
    brain = get_or_create_brain(scale_factor=scale_factor, force_new=force_new_brain)
    
    # Use ResumableSimulation instead of AdvancedSimulation
    print("2. Setting up simulation...")
    sim = ResumableSimulation(
        brain=brain,
        duration=duration,
        dt=dt,
        checkpoint_frequency=1000  # Save checkpoint every 1000 timesteps
    )
    
    # Basic structure tests
    print("\n3. Testing brain structure:")
    print(f"- Total neurons: {len(brain.neurons):,}")
    print(f"- Total connections: {len(brain.connections):,}")
    
    # Test region connectivity
    print("\n4. Testing region connectivity:")
    for region_name, region in brain.regions.items():
        # Count outgoing connections
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
    
    # Add stimulus to V1 layer 4
    try:
        v1_l4_neurons = brain.get_layer_neurons('v1', CorticalLayer.L4)
        if v1_l4_neurons:
            print(f"\n6. Stimulating {min(100, len(v1_l4_neurons))} neurons in V1 Layer 4")
            stim_neurons = v1_l4_neurons[:100]  # Stimulate first 100 neurons
            for neuron in stim_neurons:
                neuron.receive_input(-1, 2.0, 0)  # External stimulus
        else:
            print("\n6. Warning: No neurons found in V1 Layer 4")
    except Exception as e:
        print(f"\n6. Error stimulating V1 Layer 4: {str(e)}")
    
    # Save simulation ID for potential resume
    sim_id = None
    if sim.checkpoint_manager.last_checkpoint:
        sim_id = sim.checkpoint_manager._get_simulation_id(sim)
        id_file = Path("last_simulation_id.txt")
        id_file.write_text(sim_id)
        print(f"\nSimulation ID saved to {id_file}")
    
    try:
        print("\n7. Running simulation...")
        sim.run(resume_id=resume_id)
        
        # Analyze results
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
        # Spike raster plot
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
    firing_rates = np.sum(sim.spike_record, axis=0) / (sim.duration/1000.0)  # Hz
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
    parser = argparse.ArgumentParser(description='Digital Brain Audio Processing')
    parser.add_argument('--mode', choices=['test', 'audio'], default='test',
                        help='Operation mode: test or audio processing')
    parser.add_argument('--audio-file', type=str, help='Path to audio file for processing')
    parser.add_argument('--scale', type=float, default=0.001,
                        help='Brain scale factor (smaller = faster processing)')
    parser.add_argument('--resume-id', type=str, help='Resume from checkpoint ID')
    parser.add_argument('--force-new-brain', action='store_true',
                        help='Force creation of new brain instead of using cache')
    args = parser.parse_args()
    
    try:
        if args.mode == 'audio':
            if not args.audio_file:
                print("Error: --audio-file required for audio processing mode")
                sys.exit(1)
            process_audio(args.audio_file, args.scale, args.force_new_brain)
        else:
            # Original test functionality
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