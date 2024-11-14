# main.py
import numpy as np
from digital_brain import ModularDigitalBrain
from simulation import AdvancedSimulation
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
        total_connections = sum(1 for conn in brain.connections.keys() 
                              if region.start_idx <= conn[0] < region.end_idx)
        print(f"- {region_name}: {total_connections:,} outgoing connections")
    
    # Run simulation
    print("\n4. Running test simulation...")
    sim = AdvancedSimulation(brain, duration=duration, dt=dt)
    
    # Add some stimulus to V1 layer 4
    v1_l4_neurons = brain.get_layer_neurons('v1', brain.regions['v1'].layer_configs.keys()[3])
    if v1_l4_neurons:
        stim_neurons = v1_l4_neurons[:100]  # Stimulate first 100 neurons
        for neuron in stim_neurons:
            neuron.receive_input(-1, 2.0, 0)  # External stimulus
    
    sim.run()
    
    # Analyze results
    print("\n5. Analyzing simulation results:")
    
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
    print("\n6. Layer-specific activity:")
    for region_name, region in brain.regions.items():
        for layer, config in region.layer_configs.items():
            layer_start = config.start_idx
            layer_end = config.end_idx
            layer_rates = firing_rates[layer_start:layer_end]
            if len(layer_rates) > 0:
                mean_layer_rate = np.mean(layer_rates)
                print(f"- {region_name} {layer.name}: {mean_layer_rate:.2f} Hz")
    
    # Test homeostatic plasticity
    print("\n7. Testing homeostatic plasticity:")
    initial_weights = [conn.weight for conn in brain.connections.values()]
    final_weights = [conn.weight for conn in brain.connections.values()]
    weight_changes = np.array(final_weights) - np.array(initial_weights)
    print(f"- Mean weight change: {np.mean(weight_changes):.3f}")
    print(f"- Std weight change: {np.std(weight_changes):.3f}")
    
    return sim

if __name__ == "__main__":
    # Run tests with small scale factor for quick testing
    simulation = test_digital_brain(scale_factor=0.001, duration=1000.0, dt=0.1)
