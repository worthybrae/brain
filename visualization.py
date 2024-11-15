# visualization.py
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt backend instead of default
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from queue import Queue
from threading import Lock
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class RealtimeVisualizer:
    def __init__(self):
        self.data_queue = Queue()
        self.rate_history = []
        self.time_history = []
        self.lock = Lock()
        self.fig = None
        self.axes = None
        self.animation = None
        
    def setup(self):
        """Initialize visualization on the main thread."""
        self.fig = plt.figure(figsize=(15, 8))
        
        # Create subplots
        self.axes = []
        self.axes.append(plt.subplot2grid((3, 2), (0, 0), colspan=2))  # Firing rate plot
        self.axes.append(plt.subplot2grid((3, 2), (1, 0), colspan=2))  # Raster plot
        self.axes.append(plt.subplot2grid((3, 2), (2, 0)))  # Region activity
        self.axes.append(plt.subplot2grid((3, 2), (2, 1)))  # Network state
        
        self.axes[0].set_title('Mean Firing Rate')
        self.axes[1].set_title('Spike Raster')
        self.axes[2].set_title('Region Activity')
        self.axes[3].set_title('Network State')
        
        plt.tight_layout()
        
        # Start animation
        self.animation = FuncAnimation(
            self.fig, self._update_plots, interval=100,
            cache_frame_data=False
        )
        
    def _update_plots(self, frame):
        """Update all plots with latest data."""
        if self.data_queue.empty():
            return
            
        with self.lock:
            data = self.data_queue.get()
            stats = data['stats']
            
            # Update firing rate history
            self.rate_history.append(stats['mean_firing_rate'])
            self.time_history.append(data['time'])
            
            # Limit history length
            max_history = 1000
            if len(self.rate_history) > max_history:
                self.rate_history.pop(0)
                self.time_history.pop(0)
            
            # Update firing rate plot with multiple traces
            self.axes[0].clear()
            self.axes[0].plot(self.time_history, self.rate_history, 
                            label='Short-term Rate', color='blue')
            if len(self.time_history) > 0:
                self.axes[0].axhline(y=stats['instant_rate'], 
                                color='red', linestyle='--', 
                                label='Instant Rate', alpha=0.5)
            self.axes[0].set_title('Firing Rates')
            self.axes[0].set_xlabel('Time (s)')
            self.axes[0].set_ylabel('Rate (Hz)')
            self.axes[0].legend(loc='upper right')
            
            # Update raster plot
            self.axes[1].clear()
            if 'spike_record' in data:
                spikes = data['spike_record']
                if np.any(spikes):
                    times, neurons = np.where(spikes)
                    times = times * 0.1  # Convert to milliseconds
                    self.axes[1].scatter(times, neurons, s=1, c='black', alpha=0.5)
                    self.axes[1].set_xlim(0, 100)
                    
            self.axes[1].set_title('Spike Raster (last 100ms)')
            self.axes[1].set_xlabel('Time (ms)')
            self.axes[1].set_ylabel('Neuron Index')
            
            # Update region activity with detailed view
            self.axes[2].clear()
            if 'region_details' in stats:
                regions = list(stats['region_details'].keys())
                x = np.arange(len(regions))
                width = 0.35
                
                # Plot voltage-based activity
                voltages = [stats['region_details'][r]['voltage'] for r in regions]
                self.axes[2].bar(x - width/2, voltages, width, 
                            label='Voltage', color='blue', alpha=0.6)
                
                # Plot spike-based activity
                spike_rates = [stats['region_details'][r]['spike_rate'] for r in regions]
                self.axes[2].bar(x + width/2, spike_rates, width,
                            label='Spike Rate', color='red', alpha=0.6)
                
                self.axes[2].set_xticks(x)
                self.axes[2].set_xticklabels(regions)
                self.axes[2].legend()
            
            self.axes[2].set_title('Region Activity')
            plt.setp(self.axes[2].xaxis.get_majorticklabels(), rotation=45)
            
            # Update network state with enhanced metrics
            self.axes[3].clear()
            state_text = (
                f"State: {stats['network_state']}\n"
                f"Sync Index: {stats['synchrony_index']:.2f}\n"
                f"Instant Sync: {stats['instant_sync']:.2f}\n"
                f"Active Neurons: {stats['active_neurons']}\n"
                f"({stats['active_fraction']*100:.1f}% of network)\n"
                f"Timestep: {stats['timestep']}"
            )
            self.axes[3].text(0.5, 0.5, state_text,
                            horizontalalignment='center',
                            verticalalignment='center')
            self.axes[3].axis('off')
            
            plt.tight_layout()
    
    def update_data(self, data):
        """Thread-safe method to add new data."""
        self.data_queue.put(data)
    
    def start(self):
        """Start visualization on main thread."""
        plt.show()
    
    def stop(self):
        """Clean up resources."""
        if self.animation:
            self.animation.event_source.stop()
        plt.close(self.fig)
