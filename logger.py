# logger.py
import logging
import os
from datetime import datetime
from typing import Optional
import psutil

class BrainLogger:
    _instance: Optional['BrainLogger'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BrainLogger, cls).__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance
    
    def _initialize_logger(self):
        """Initialize the logger with file and console handlers"""
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        # Generate timestamp for log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'logs/brain_{timestamp}.log'
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Initialize logger
        self.logger = logging.getLogger('BrainLogger')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler (all levels)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler (INFO and above)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logger initialized. Log file: {log_file}")
    
    @staticmethod
    def get_logger():
        """Get the singleton logger instance"""
        if BrainLogger._instance is None:
            BrainLogger()
        return BrainLogger._instance.logger
    
    def log_memory_usage(self):
        """Log current memory usage"""
        try:
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
            self.logger.info(f"Current memory usage: {memory_usage:.2f} MB")
        except ImportError:
            self.logger.warning("psutil not installed. Cannot log memory usage.")
    
    def log_brain_stats(self, brain):
        """Log statistics about the brain's current state"""
        stats = {
            "Total neurons": brain.total_neurons,
            "Total connections": len(brain.connection_properties),
            "Regions": len(brain.regions),
        }
        
        self.logger.info("Brain Statistics:")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value:,}")