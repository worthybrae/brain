import numpy as np
from digital_brain import ModularDigitalBrain
from simulation import AdvancedSimulation
from cortical_layers import CorticalLayer
import matplotlib.pyplot as plt

def test_digital_brain(scale_factor=0.001, duration=1000.0, dt=0.1):
    """
    Comprehensive test of the digital brain implementation.
    
    Args:
        scale_factor: Brain size scaling factor (smaller = faster testing)
        duration: Simulation duration in ms
        dt: Integration time step in ms
    """
    print("\n=== Digital Brain Test Suite ===\n")
    
    # Initialize brain with small scale for testing
    print("1. Initializing brain...")
    brain = ModularDigitalBrain(scale_factor=scale_factor)
    
    # Basic structure tests
    print("\n2. Testing brain structure:")
    print(f"- Total neurons: {len(brain.neurons):,}")
    print(f"- Total connections: {len(brain.connections):,}")
    
    # Test region connectivity
    print("\n3. Testing region connectivity:")
    for region_name, region in brain.regions.items():
        # Count outgoing connections
        outgoing = 0
        for (src, _) in brain.connections.keys():
            if any(layer_config.start_idx <= src < layer_config.end_idx 
                  for layer_config in region.layer_configs.values()):
                outgoing += 1
        print(f"- {region_name}: {outgoing:,} outgoing connections")
    
    # Run simulation
    print("\n4. Running test simulation...")
    sim = AdvancedSimulation(brain, duration=duration, dt=dt)
    
    # Add stimulus to V1 layer 4
    try:
        v1_l4_neurons = brain.get_layer_neurons('v1', CorticalLayer.L4)
        if v1_l4_neurons:
            print(f"- Stimulating {min(100, len(v1_l4_neurons))} neurons in V1 Layer 4")
            stim_neurons = v1_l4_neurons[:100]  # Stimulate first 100 neurons
            for neuron in stim_neurons:
                neuron.receive_input(-1, 2.0, 0)  # External stimulus
        else:
            print("- Warning: No neurons found in V1 Layer 4")
    except Exception as e:
        print(f"- Error stimulating V1 Layer 4: {str(e)}")
    
    # Print layer information
    print("\n5. Layer configuration:")
    for region_name, region in brain.regions.items():
        print(f"\n{region_name} layers:")
        for layer, config in region.layer_configs.items():
            neuron_count = config.end_idx - config.start_idx
            print(f"- {layer.name}: {neuron_count} neurons ({config.start_idx}-{config.end_idx})")
    
    try:
        sim.run()
        
        # Analyze results
        print("\n6. Analyzing simulation results:")
        
        # Plot spike raster
        active_neurons = np.where(np.sum(sim.spike_record, axis=0) > 0)[0]
        if len(active_neurons) > 0:
            plt.figure(figsize=(12, 6))
            spike_times = np.where(sim.spike_record[:, active_neurons])[0]
            spike_neurons = np.where(sim.spike_record[:, active_neurons])[1]
            plt.scatter(spike_times * dt, spike_neurons, s=1, c='black')
            plt.xlabel('Time (ms)')
            plt.ylabel('Neuron Index')
            plt.title('Spike Raster Plot')
            plt.show()
        else:
            print("- Warning: No active neurons detected")
        
        # Calculate firing rates
        firing_rates = np.sum(sim.spike_record, axis=0) / (duration/1000.0)  # Hz
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
        
        # Analyze layer-specific activity
        print("\n7. Layer-specific activity:")
        for region_name, region in brain.regions.items():
            print(f"\n{region_name}:")
            for layer, config in region.layer_configs.items():
                layer_rates = firing_rates[config.start_idx:config.end_idx]
                if len(layer_rates) > 0:
                    active_count = np.sum(layer_rates > 0)
                    mean_layer_rate = np.mean(layer_rates)
                    print(f"- {layer.name}: {mean_layer_rate:.2f} Hz "
                          f"({active_count} active neurons)")
        
        # Connection weight analysis
        print("\n8. Connection analysis:")
        weights = [conn.weight for conn in brain.connections.values()]
        print(f"- Mean weight: {np.mean(weights):.3f}")
        print(f"- Weight std: {np.std(weights):.3f}")
        print(f"- Min weight: {np.min(weights):.3f}")
        print(f"- Max weight: {np.max(weights):.3f}")
        
        return sim
        
    except Exception as e:
        print(f"\nError during simulation: {str(e)}")
        raise

if __name__ == "__main__":
    # Run tests with small scale factor for quick testing
    simulation = test_digital_brain(scale_factor=0.001, duration=1000.0, dt=0.1)
    