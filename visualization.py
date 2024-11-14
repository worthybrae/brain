# visualization.py
import matplotlib.pyplot as plt
import numpy as np

# visualization.py
def plot_raster(simulation, num_neurons=100):
    """Plot a raster plot of neural activity."""
    activity = np.array(simulation.activity_record)
    times, neurons = np.nonzero(activity[:, :num_neurons])
    plt.figure(figsize=(12, 6))
    plt.scatter(times, neurons, s=1)
    plt.xlabel('Time Steps')
    plt.ylabel('Neuron Index')
    plt.title('Raster Plot of Neural Activity')
    plt.show()

