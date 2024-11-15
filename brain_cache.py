# brain_cache.py
import pickle
from pathlib import Path
from typing import Optional
import os
import hashlib
from digital_brain import ModularDigitalBrain

class BrainCache:
    """Handles caching and loading of ModularDigitalBrain instances."""
    
    def __init__(self, cache_dir: str = "brain_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _generate_cache_key(self, scale_factor: float) -> str:
        """Generate a unique cache key based on brain parameters."""
        # Add more parameters here if they affect brain structure
        params = f"scale_factor={scale_factor}"
        return hashlib.md5(params.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cached brain."""
        return self.cache_dir / f"brain_{cache_key}.pkl"
    
    def save_brain(self, brain: ModularDigitalBrain) -> str:
        """Save brain to cache and return cache key."""
        cache_key = self._generate_cache_key(brain.scale_factor)
        cache_path = self._get_cache_path(cache_key)
        
        # Save brain to file
        with open(cache_path, 'wb') as f:
            pickle.dump(brain, f)
            
        return cache_key
    
    def load_brain(self, scale_factor: float) -> Optional[ModularDigitalBrain]:
        """Load brain from cache if available."""
        cache_key = self._generate_cache_key(scale_factor)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cached brain: {e}")
                # Remove corrupted cache file
                os.remove(cache_path)
        return None
    
    def clear_cache(self):
        """Clear all cached brains."""
        for cache_file in self.cache_dir.glob("brain_*.pkl"):
            os.remove(cache_file)

def get_or_create_brain(scale_factor: float = 0.001, force_new: bool = False) -> ModularDigitalBrain:
    """Get cached brain or create new one if needed."""
    cache = BrainCache()
    
    if not force_new:
        # Try to load from cache
        cached_brain = cache.load_brain(scale_factor)
        if cached_brain is not None:
            print("Using cached brain configuration")
            return cached_brain
    
    # Create new brain
    print("Creating new brain configuration...")
    brain = ModularDigitalBrain(scale_factor=scale_factor)
    
    # Save to cache
    if not force_new:
        cache.save_brain(brain)
        print("Brain configuration cached for future use")
    
    return brain