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

    # Configure neural network - simplified architecture
    input_features = 9
    neural_network = NeuralNetwork(
        layers=[input_features, 16, 8, 1],  # Simplified network
        learning=0.001,
        beta=0.9
    )

    # Initialize betting system
    betting_system = NFLBettingSystem(
        feature_processor=feature_processor,
        neural_network=neural_network,
        initial_bankroll=10000.0
    )

    # Process features
    print("Processing features...")
    processed_data = betting_system.process_game_data(data)

    # Normalize the target variable (spread)
    spread_mean = processed_data['spread_favorite'].mean()
    spread_std = processed_data['spread_favorite'].std()
    normalized_spread = (processed_data['spread_favorite'] - spread_mean) / spread_std

    # Split data into training and testing sets
    train_size = int(len(processed_data) * 0.8)
    train_data = processed_data[:train_size]
    test_data = processed_data[train_size:]

    # Reset index of test_data
    test_data = test_data.reset_index(drop=True)

    # Prepare training data with normalization
    print("Preparing training data...")
    X_train = betting_system._prepare_prediction_features(train_data)
    y_train = normalized_spread[:train_size].values.reshape(-1, 1)

    # Standardize features
    feature_means = X_train.mean(axis=0)
    feature_stds = X_train.std(axis=0)
    X_train = (X_train - feature_means) / feature_stds

    # Train the model
    print("Training neural network...")
    neural_network.train(X_train, y_train, epochs=2000, batchsize=32)

    # Prepare test data
    print("\nTesting betting system...")
    X_test = betting_system._prepare_prediction_features(test_data)
    X_test = (X_test - feature_means) / feature_stds

    # Get predictions and denormalize
    raw_predictions = betting_system.predict_spread(test_data)
    predictions = (raw_predictions * spread_std) + spread_mean

    print("\nPrediction Statistics:")
    print(f"Mean predicted spread: {predictions.mean():.2f}")
    print(f"Std predicted spread: {predictions.std():.2f}")
    print(f"Min predicted spread: {predictions.min():.2f}")
    print(f"Max predicted spread: {predictions.max():.2f}")

    # Find value bets
    opportunities = betting_system.find_value_bets(
        test_data,
        predictions,
        minimum_edge=4.0  # Using our new conservative threshold
    )

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

        # Print sample opportunities with more detail
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