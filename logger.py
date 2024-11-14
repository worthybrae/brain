# logger.py
import logging
import os
from datetime import datetime
from typing import Optional


class BrainLogger:
    _instance: Optional['BrainLogger'] = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BrainLogger, cls).__new__(cls)
        return cls._instance
    
    def _initialize_logger(self):
        """Initialize the logger with file and console handlers"""
        if self._initialized:
            return
            
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
        if not self.logger.handlers:  # Only add handlers if none exist
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
        
        self._initialized = True
    
    @staticmethod
    def get_logger():
        """Get the singleton logger instance"""
        if BrainLogger._instance is None:
            BrainLogger()
        if not BrainLogger._instance._initialized:
            BrainLogger._instance._initialize_logger()
        return BrainLogger._instance.logger