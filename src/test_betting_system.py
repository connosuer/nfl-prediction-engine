import pandas as pd
import numpy as np
from betting_system import NFLBettingSystem
from featureEngineering import NFLFeatureProcessor
from nn import NeuralNetwork


def test_betting_system():
    # Load historical data
    print("Loading historical data...")
    data = pd.read_csv('../data/historic_clean.csv')

    # Initialize components
    print("Initializing system components...")
    feature_processor = NFLFeatureProcessor()

    # Configure feature columns
    feature_cols = [
        'point_differential', 'total_points', 'over_under_performance', 'spread_performance',
        'home_last3_points', 'home_last3_points_allowed', 'home_cover_rate', 'home_streak',
        'away_last3_points', 'away_last3_points_allowed', 'away_cover_rate', 'away_streak',
        'power_rating_diff', 'is_home_favorite'
    ]

    # Process features
    print("Processing features...")
    processed_data = feature_processor.process_initial_features(data)

    # Prepare features and target variable
    X = processed_data[feature_cols].values
    y = processed_data['spread_favorite'].values.reshape(-1, 1)

    # Normalize the target variable (spread)
    spread_mean = y.mean()
    spread_std = y.std()
    y_norm = (y - spread_mean) / spread_std

    # Split data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train = X[:train_size]
    y_train = y_norm[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    # Normalize features
    feature_means = X_train.mean(axis=0)
    feature_stds = X_train.std(axis=0)
    feature_stds[feature_stds == 0] = 1.0  # Prevent division by zero

    X_train = (X_train - feature_means) / feature_stds
    X_test = (X_test - feature_means) / feature_stds

    # Configure neural network
    input_features = X_train.shape[1]
    nn = NeuralNetwork(
        layers=[input_features, 32, 16, 1],
        activation='relu',
        output_activation='linear',
        optimizer='momentum',
        learning=0.001,
        beta=0.9
    )

    # Train the neural network
    print("Training neural network...")
    nn.train(X_train, y_train, epochs=1000, batchsize=32)

    # Predict spreads on the test set
    print("\nEvaluating neural network...")
    raw_predictions = nn.prediction(X_test)
    predictions = (raw_predictions * spread_std) + spread_mean  # Denormalize predictions

    # Evaluation metrics
    mse = np.mean((predictions - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y_test))
    print("\nEvaluation Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Initialize betting system
    betting_system = NFLBettingSystem(
        feature_processor=feature_processor,
        neural_network=nn,
        initial_bankroll=10000.0
    )

    # Find value bets
    print("\nFinding value bets...")
    test_data = processed_data[train_size:].reset_index(drop=True)  # Ensure indices match
    opportunities = betting_system.find_value_bets(
        test_data,
        predictions,
        minimum_edge=4.0  # Conservative threshold
    )

    # Print betting results
    print(f"\nFound {len(opportunities)} potential betting opportunities")
    if len(opportunities) > 0:
        total_pnl = 0
        winning_bets = 0
        total_stake = 0

        for opp in opportunities:
            game_data = test_data.loc[opp['game_id']]
            betting_system.execute_trade(opp)

            pnl = betting_system.evaluate_position(
                opp,
                actual_score_home=game_data['score_home'],
                actual_score_away=game_data['score_away']
            )

            total_pnl += pnl
            total_stake += opp['recommended_stake']
            if pnl > 0:
                winning_bets += 1

        # Enhanced results reporting
        print("\nBacktesting Results:")
        print(f"Total Bets: {len(opportunities)}")
        print(f"Winning Bets: {winning_bets}")
        print(f"Win Rate: {(winning_bets / len(opportunities) * 100):.2f}%")
        print(f"Total P&L: ${total_pnl:.2f}")
        print(f"Total Stake: ${total_stake:.2f}")
        print(f"ROI: {(total_pnl / total_stake * 100):.2f}%")
        print(f"Average Bet Size: ${total_stake / len(opportunities):.2f}")

        # Sample betting opportunities
        print("\nSample Betting Opportunities:")
        for opp in opportunities[:5]:
            print(f"\nDate: {opp['date']}")
            print(f"Game: {opp['home_team']} vs {opp['away_team']}")
            print(f"Market Spread: {opp['market_spread']:.1f}")
            print(f"Predicted Spread: {opp['predicted_spread']:.1f}")
            print(f"Edge: {opp['edge']:.1f} points")
            print(f"Bet Side: {opp['bet_side']}")
            print(f"Recommended Stake: ${opp['recommended_stake']:.2f}")


if __name__ == "__main__":
    test_betting_system()