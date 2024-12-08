import pandas as pd
from featureEngineering import NFLFeatureProcessor


def test_feature_engineering():
    df = pd.read_csv('../data/historic_clean.csv')

    processor = NFLFeatureProcessor()

    try:
        processed_df = processor.process_initial_features(df)

        print("\nProcessed Data Info:")
        print(f"Number of rows: {len(processed_df)}")
        print(f"Number of columns: {len(processed_df.columns)}")

        original_cols = set(df.columns)
        new_cols = set(processed_df.columns) - original_cols
        print("\nNew features created:")
        for col in sorted(new_cols):
            print(f"- {col}")

        print("\nSample of processed data (first 5 rows, new features only):")
        print(processed_df[list(new_cols)].head())

        print("\nSummary statistics for key features:")
        key_features = ['total_points', 'point_differential', 'spread_performance',
                        'over_under_performance', 'power_rating_diff']
        print(processed_df[key_features].describe())

        return processed_df

    except Exception as e:
        print(f"Error processing features: {str(e)}")
        raise


if __name__ == "__main__":
    processed_data = test_feature_engineering()