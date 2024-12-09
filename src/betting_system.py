from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from featureEngineering import NFLFeatureProcessor
from nn import NeuralNetwork


class NFLBettingSystem:
    def __init__(self,
                 feature_processor: NFLFeatureProcessor,
                 neural_network: NeuralNetwork,
                 initial_bankroll: float = 10000.0):
        self.feature_processor = feature_processor
        self.neural_network = neural_network
        self.bankroll = initial_bankroll
        self.active_positions: List[Dict] = []

    def process_game_data(self, game_data: pd.DataFrame) -> pd.DataFrame:
        """Process raw game data into features for prediction"""
        return self.feature_processor.process_initial_features(game_data)

    def predict_spread(self, features: pd.DataFrame) -> np.ndarray:
        """Generate spread predictions using the neural network"""
        prediction_features = self._prepare_prediction_features(features)
        return self.neural_network.prediction(prediction_features)

    def _prepare_prediction_features(self, features: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix for neural network input"""
        selected_features = [
            'power_rating_diff',
            'home_last3_points', 'home_last3_points_allowed',
            'away_last3_points', 'away_last3_points_allowed',
            'home_cover_rate', 'away_cover_rate',
            'home_streak', 'away_streak'
        ]
        X = features[selected_features].values
        return X

    def find_value_bets(self,
                        features: pd.DataFrame,
                        predicted_spreads: np.ndarray,
                        minimum_edge: float = 2.0) -> List[Dict]:
        """
        Identify betting opportunities where our prediction differs significantly from the market

        Args:
            features: Processed feature DataFrame
            predicted_spreads: Our model's spread predictions
            minimum_edge: Minimum point difference to consider a value bet

        Returns:
            List of dictionaries containing betting opportunities
        """
        opportunities = []
        if len(predicted_spreads.shape) > 1:
            predicted_spreads = predicted_spreads.flatten()

        for idx, row in features.iterrows():
            market_spread = row['spread_favorite'] if row['team_favorite_id'] == row['team_home'] else -row[
                'spread_favorite']
            predicted_spread = predicted_spreads[idx]

            # Calculate edge (difference between our prediction and market)
            edge = abs(predicted_spread - market_spread)

            if edge >= minimum_edge:
                bet_side = 'home' if predicted_spread > market_spread else 'away'
                opportunities.append({
                    'game_id': idx,
                    'date': row['schedule_date'],
                    'home_team': row['team_home'],
                    'away_team': row['team_away'],
                    'market_spread': market_spread,
                    'predicted_spread': predicted_spread,
                    'edge': edge,
                    'bet_side': bet_side,
                    'recommended_stake': self._calculate_stake(edge)
                })

        return opportunities

    def _calculate_stake(self, edge: float) -> float:
        """
        Calculate recommended stake size based on edge and bankroll
        Using a modified Kelly Criterion
        """
        # Assumed win probability based on edge
        # This is a simplified approach - could be made more sophisticated
        base_prob = 0.5
        edge_factor = edge / 10  # Scale edge to probability adjustment
        win_prob = min(base_prob + edge_factor, 0.75)  # Cap at 75% probability

        # Kelly fraction calculation (using -110 as standard odds)
        b = 0.909  # decimal odds minus 1 for standard -110 odds
        q = 1 - win_prob
        f = (win_prob * b - q) / b

        # Use quarter Kelly for more conservative sizing
        conservative_f = f * 0.25

        return round(self.bankroll * conservative_f, 2)

    def evaluate_position(self, position: Dict, actual_score_home: float, actual_score_away: float) -> float:
        """
        Evaluate the P&L of a position based on actual game results

        Returns:
            Profit/loss amount
        """
        actual_spread = actual_score_home - actual_score_away

        # Determine if bet won
        if position['bet_side'] == 'home':
            won_bet = actual_spread > position['market_spread']
        else:
            won_bet = actual_spread < position['market_spread']

        # Calculate P&L (assuming -110 odds)
        if won_bet:
            return position['recommended_stake'] * 0.909  # Standard payout at -110 odds
        else:
            return -position['recommended_stake']

    def execute_trade(self, trade_info: Dict) -> None:
        """Execute a trade and update positions"""
        if trade_info['recommended_stake'] <= self.bankroll:
            self.bankroll -= trade_info['recommended_stake']
            # Store exact same dictionary to maintain consistency
            self.active_positions.append(trade_info)
