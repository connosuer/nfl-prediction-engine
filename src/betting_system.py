
# Imports
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from decimal import Decimal

from featureEngineering import NFLFeatureProcessor
from lineGenerationNN import NeuralNetwork

class NFLBettingSystem:
    def __init__(self,
                 feature_processor: NFLFeatureProcessor,
                 neural_network: NeuralNetwork,
                 initial_bankroll: float = 10000.0):
        self.feature_processor = feature_processor
        self.neural_network = neural_network
        self.bankroll = initial_bankroll
        self.active_positions: List[Dict] = []