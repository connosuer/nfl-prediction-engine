from typing import Dict, List
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

    def predict_spread(self, features: pd.DataFrame) -> np.ndarray:
        """Generate spread predictions using the neural network with double clipping"""
        prediction_features = self._prepare_prediction_features(features)
        raw_predictions = self.neural_network.prediction(prediction_features)

        # First clip after initial prediction
        clipped_predictions = np.clip(raw_predictions, -14, 14)

        # sanity checks
        mean_spread = np.mean(clipped_predictions)
        if abs(mean_spread) > 7:  # If predictions seem biased
            clipped_predictions -= mean_spread  # Center around zero
            clipped_predictions = np.clip(clipped_predictions, -14, 14)  # Clip again

        return clipped_predictions

    def find_value_bets(self,
                        features: pd.DataFrame,
                        predicted_spreads: np.ndarray,
                        minimum_edge: float = 4.0) -> List[Dict]:
        """Identify betting opportunities with double validation"""
        opportunities = []
        if len(predicted_spreads.shape) > 1:
            predicted_spreads = predicted_spreads.flatten()

        # Final clip before edge calculation
        predicted_spreads = np.clip(predicted_spreads, -14, 14)

        # Calculate average market spread for comparison
        market_spreads = []
        for _, row in features.iterrows():
            market_spread = row['spread_favorite'] if row['team_favorite_id'] == row['team_home'] else -row[
                'spread_favorite']
            market_spreads.append(market_spread)

        avg_market = np.mean(market_spreads)

        for idx, row in features.iterrows():
            market_spread = market_spreads[idx]
            predicted_spread = predicted_spreads[idx]

            # Additional sanity check - avoid systematic bias
            if abs(predicted_spread - avg_market) > 14:
                continue

            edge = predicted_spread - market_spread
            if self._is_valid_bet(edge, market_spread, row):
                bet_side = 'home' if predicted_spread > market_spread else 'away'
                stake = self._calculate_stake(abs(edge))

                opportunities.append({
                    'game_id': idx,
                    'date': row['schedule_date'],
                    'home_team': row['team_home'],
                    'away_team': row['team_away'],
                    'market_spread': market_spread,
                    'predicted_spread': predicted_spread,
                    'edge': abs(edge),
                    'bet_side': bet_side,
                    'recommended_stake': stake
                })

        return opportunities

    def _is_valid_bet(self, edge: float, market_spread: float, row: pd.Series) -> bool:
        """
        More conservative validation checks for potential bets
        """
        # so must have significant edge but not unrealistic
        if abs(edge) < 4.0 or abs(edge) > 10.0:  # Cap maximum edge
            return False

        # More conservative limits on spread ranges
        if abs(market_spread) > 7:
            return abs(edge) >= 5.0 and abs(edge) <= 8.0  # Tighter range for big spreads

        # No bets on huge spreads
        if abs(market_spread) > 10:
            return False

        # Check team form (basic)
        betting_on_home = edge > 0
        if betting_on_home and row['home_last3_points'] < row['home_last3_points_allowed']:
            return False
        elif not betting_on_home and row['away_last3_points'] < row['away_last3_points_allowed']:
            return False

        # Check reasonable scoring ranges
        if row['home_last3_points'] > 40 or row['away_last3_points'] > 40:
            return False  # Avoid betting after unusual scoring games

        return True

    def _calculate_stake(self, edge: float) -> float:
        """
        Simplified stake calculation
        """
        # Base stake is 2% of bankroll
        base_stake = self.bankroll * 0.02

        # Adjust based on edge (up to 2.5x for large edges)
        edge_multiplier = min(1 + (edge - 4) * 0.1, 2.5)

        stake = base_stake * edge_multiplier

        # Hard limits
        min_stake = 100
        max_stake = self.bankroll * 0.05

        return round(min(max(stake, min_stake), max_stake), 2)

    def evaluate_position(self, position: Dict, actual_score_home: float, actual_score_away: float) -> float:
        """
        Evaluate the P&L of a position based on actual game results
        """
        actual_spread = actual_score_home - actual_score_away

        # Determine if bet won
        if position['bet_side'] == 'home':
            won_bet = actual_spread > position['market_spread']
        else:
            won_bet = actual_spread < position['market_spread']

        # Calculate P&L (assuming -110 odds)
        if won_bet:
            return position['recommended_stake'] * 0.909
        else:
            return -position['recommended_stake']

    def execute_trade(self, trade_info: Dict) -> None:
        """
        Execute a trade and update positions
        """
        if trade_info['recommended_stake'] <= self.bankroll:
            self.bankroll -= trade_info['recommended_stake']
            self.active_positions.append(trade_info)
