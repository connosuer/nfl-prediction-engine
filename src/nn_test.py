import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from featureEngineering import NFLFeatureProcessor
from nn import NeuralNetwork


def nn_test():
    # Load Data
    data = pd.read_csv('../data/historic_clean.csv')

    # Run data through feature engineering system
    feature_processor = NFLFeatureProcessor()
    processed_data = feature_processor.process_initial_features(data)

    # Set features
    feature_cols = [
        'point_differential', 'total_points', 'over_under_performance', 'spread_performance',
        'home_last3_points', 'home_last3_points_allowed', 'home_cover_rate', 'home_streak',
        'away_last3_points', 'away_last3_points_allowed', 'away_cover_rate', 'away_streak',
        'power_rating_diff', 'is_home_favorite'
    ]

    # Set indicators and target
    X = processed_data[feature_cols].values
    y = processed_data['spread_favorite'].values.reshape(-1, 1)

    # normalize
    spread_mean = y.mean()
    spread_std = y.std()
    y_norm = (y - spread_mean) / spread_std

    # 80/20 train-test data split
    train_size = int(len(X) * 0.8)
    X_train = X[:train_size]
    y_train = y_norm[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    # standardize values
    feature_means = X_train.mean(axis=0)
    feature_sd = X_train.std(axis=0)
    feature_sd[feature_sd == 0] = 1.0

    X_train = (X_train - feature_means) / feature_sd
    X_test = (X_test - feature_means) / feature_sd

    # initialize network
    input_features = X_train.shape[1]
    nn = NeuralNetwork(
        layers=[input_features, 32, 16, 1],
        activation='relu',
        output_activation='linear',
        optimizer='momentum',
        learning=0.001,
        beta=0.9
    )

    # train
    cost_vals = nn.train(X_train, y_train, epochs=1000, batchsize=32)

    # setup data for cost plot
    filtered_epochs_10 = range(0, len(cost_vals), 10)
    filtered_costs_10 = [cost_vals[epoch] for epoch in filtered_epochs_10]

    coefficients = np.polyfit(filtered_epochs_10, filtered_costs_10, deg=1)
    trend_line = np.poly1d(coefficients)

    trend_x = np.linspace(min(filtered_epochs_10), max(filtered_epochs_10), 500)
    trend_y = trend_line(trend_x)

    # Plot cost and trend line
    plt.figure(figsize=(8, 6))
    plt.plot(filtered_epochs_10, filtered_costs_10, 'o-', label='Training Cost')
    plt.plot(trend_x, trend_y, 'r--', label='Trend')
    plt.title('Cost Over Time (Every 10 Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.legend()
    plt.show()

    filtered_epochs_100 = range(0, len(cost_vals), 100)
    filtered_costs_100 = [cost_vals[epoch] for epoch in filtered_epochs_100]

    coefficients = np.polyfit(filtered_epochs_100, filtered_costs_100, deg=1)
    trend_line = np.poly1d(coefficients)

    trend_x = np.linspace(min(filtered_epochs_100), max(filtered_epochs_100), 500)
    trend_y = trend_line(trend_x)

    plt.figure(figsize=(8, 6))
    plt.plot(filtered_epochs_100, filtered_costs_100, 'o-', label='Training Cost')
    plt.plot(trend_x, trend_y, 'r--', label='Trend')
    plt.title('Cost Over Time (Every 100 Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.legend()
    plt.show()

    # evaluate and normalize
    y_pred_norm = nn.prediction(X_test)
    y_pred = (y_pred_norm * spread_std) + spread_mean
    mse = np.mean((y_pred - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_test))
    print("\nEvaluation on Test Set:")
    print(f"RMSE: {rmse:.4f}")

    # Sample predictions vs actual
    print("\nSample Predictions vs Actual:")
    for i in range(min(10, len(y_test))):
        print(f"Predicted: {y_pred[i, 0]:.2f}, Actual: {y_test[i, 0]:.2f}")


if __name__ == "__main__":
    nn_test()