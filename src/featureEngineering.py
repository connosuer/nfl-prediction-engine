import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class NFLFeatureProcessor:
    def __init__(self):
        self.team_ratings = {}

        # map of the team names and standardizing them into abbr
        self.team_map = {
            'New Orleans Saints': 'NO',
            'Minnesota Vikings': 'MIN',
            'Buffalo Bills': 'BUF',
            'Miami Dolphins': 'MIA',
            'Chicago Bears': 'CHI',
            'Detroit Lions': 'DET',
            'Houston Texans': 'HOU',
            'Indianapolis Colts': 'IND',
            'Jacksonville Jaguars': 'JAX',
            'Denver Broncos': 'DEN',
            'New England Patriots': 'NE',
            'Cincinnati Bengals': 'CIN',
            'New York Giants': 'NYG',
            'Carolina Panthers': 'CAR',
            'Philadelphia Eagles': 'PHI',
            'Green Bay Packers': 'GB',
            'Pittsburgh Steelers': 'PIT',
            'Atlanta Falcons': 'ATL',
            'Seattle Seahawks': 'SEA',
            'San Francisco 49ers': 'SF',
            'St. Louis Rams': 'STL',
            'Arizona Cardinals': 'ARI',
            'Tampa Bay Buccaneers': 'TB',
            'Cleveland Browns': 'CLE',
            'Tennessee Titans': 'TEN',
            'Oakland Raiders': 'OAK',
            'Washington Football Team': 'WAS',
            'Washington Redskins': 'WAS',
            'Dallas Cowboys': 'DAL',
            'Kansas City Chiefs': 'KC',
            'San Diego Chargers': 'SD',
            'New York Jets': 'NYJ',
            'Baltimore Ravens': 'BAL',
            'Los Angeles Rams': 'LAR',
            'Los Angeles Chargers': 'LAC',
            'Las Vegas Raiders': 'LVR',
            'Washington Commanders': 'WAS'
        }

    def process_initial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        core predictive features
        orchestrating by calling helper methods

        Args:
            df: Input DataFrame containing NFL game data

        Returns:
            DataFrame with engineered features
        """
        processed = df.copy()

        # validating all required columns
        required_columns = {
            'schedule_date', 'team_home', 'team_away', 'score_home', 'score_away',
            'team_favorite_id', 'spread_favorite', 'over_under_line'
        }
        # else if column doesnt exist
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # calling all helper methods
        processed = self._add_basic_features(processed)
        processed = self._add_team_performance(processed)
        processed = self._add_power_ratings(processed)
        return processed

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add fundamental game-level features with corrected spread calculation
        """
        df = df.copy()

        # Basic score features

        # summing the scores of home and away team
        df['total_points'] = df['score_home'] + df['score_away']
        # difference of scores
        df['point_differential'] = df['score_home'] - df['score_away']
        # binary flag: is home team the favorite
        df['is_home_favorite'] = (df['team_favorite_id'] == df['team_home']).astype(int)

        # spread performance calculation
        df['spread_performance'] = df.apply(lambda row: self._calculate_spread_performance(row), axis=1)

        # measuring total points comparison to line
        df['over_under_performance'] = df['total_points'] - df['over_under_line']
        # did favorite cover the spread
        df['favorite_won'] = (df['spread_performance'] > 0).astype(int)

        # validating spread calculations
        self._validate_spread_calculations(df)

        return df

    def _calculate_spread_performance(self, row) -> float:
        """
        Calculate spread performance with standardized team names
        """

        # mapping the home to corresponding abbr
        home_team_abbrev = self.team_map.get(row['team_home'])
        # if is null then print
        if home_team_abbrev is None:
            print(f"Warning: Unknown team name: {row['team_home']}")
            home_team_abbrev = row['team_home']

        # is home team favorite
        is_home_favorite = (row['team_favorite_id'] == home_team_abbrev)
        # spread perf based on differential and spread
        point_diff = row['point_differential']
        spread = abs(row['spread_favorite'])

        if is_home_favorite:
            # Home favorite: actual margin vs spread
            return point_diff - spread
        else:
            # Away favorite: actual margin vs spread
            return -point_diff - spread

    def _validate_spread_calculations(self, df: pd.DataFrame) -> None:
        """
        Validate spread calculations with team name
        """
        print("\nSpread Calculation Examples:")
        for _, row in df.head().iterrows():
            home_team = row['team_home']
            away_team = row['team_away']
            home_abbrev = self.team_map.get(home_team, home_team)
            away_abbrev = self.team_map.get(away_team, away_team)

            print(f"\nGame: {home_team} ({home_abbrev}) vs {away_team} ({away_abbrev})")
            print(f"Score: {row['score_home']} - {row['score_away']}")
            print(f"Spread: {row['spread_favorite']} (Favorite: {row['team_favorite_id']})")
            print(f"Is Home Favorite: {row['team_favorite_id'] == home_abbrev}")
            print(f"Point Differential: {row['point_differential']}")

            if row['team_favorite_id'] == home_abbrev:
                print(
                    f"Home favorite calculation: {row['point_differential']} - {abs(row['spread_favorite'])} = {row['spread_performance']}")
            else:
                print(
                    f"Away favorite calculation: -({row['point_differential']}) - {abs(row['spread_favorite'])} = {row['spread_performance']}")

        # market efficiency stats
        fav_cover_rate = (df['spread_performance'] > 0).mean() * 100
        push_rate = (df['spread_performance'] == 0).mean() * 100
        dog_cover_rate = (df['spread_performance'] < 0).mean() * 100

        print("\nMarket Efficiency Stats:")
        print(f"Favorite cover rate: {fav_cover_rate:.1f}%")
        print(f"Push rate: {push_rate:.1f}%")
        print(f"Underdog cover rate: {dog_cover_rate:.1f}%")

    def _add_team_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate team-specific performance metrics
        """
        df = df.copy()

        # ensuring that date is in right format
        if not pd.api.types.is_datetime64_any_dtype(df['schedule_date']):
            df['schedule_date'] = pd.to_datetime(df['schedule_date'])

        df = df.sort_values('schedule_date')

        for team_type in ['home', 'away']:
            opp_type = 'away' if team_type == 'home' else 'home'
            team_groups = df.groupby(f'team_{team_type}')

            # rolling calculations

            # averages of point scored
            df[f'{team_type}_last3_points'] = team_groups[f'score_{team_type}'].transform(
                lambda x: x.rolling(3, min_periods=1).mean().fillna(0)
            )
            # averages of points allowed
            df[f'{team_type}_last3_points_allowed'] = team_groups[f'score_{opp_type}'].transform(
                lambda x: x.rolling(3, min_periods=1).mean().fillna(0)
            )
            # cover rate of the last 5 games
            df[f'{team_type}_cover_rate'] = team_groups['spread_performance'].transform(
                lambda x: x.rolling(5, min_periods=1).apply(lambda x: (x > 0).mean()).fillna(0.5)
            )
            # win streak after last three games
            df[f'{team_type}_streak'] = team_groups['point_differential'].transform(
                lambda x: x.rolling(3, min_periods=1).apply(lambda x: sum(x > 0)).fillna(0)
            )

        return df

    def _add_power_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate and add power ratings with corrected implementation
        """
        df = df.copy()

        # Initialize ratings if empty
        if not self.team_ratings:
            teams = pd.concat([df['team_home'], df['team_away']]).unique()
            self.team_ratings = {team: 0.0 for team in teams}

        # Calculate initial power rating differences
        df['power_rating_diff'] = df.apply(
            lambda row: self.team_ratings[row['team_home']] - self.team_ratings[row['team_away']],
            axis=1
        )

        # Update ratings iteratively through season
        learning_rate = 0.15

        # Sort by date to ensure chronological processing
        df = df.sort_values('schedule_date')

        # creating temp columns to store updated ratings
        new_ratings = self.team_ratings.copy()

        for _, row in df.iterrows():
            home_team = row['team_home']
            away_team = row['team_away']
            point_diff = row['point_differential']

            # Calculate expected margin based on current ratings
            expected_margin = new_ratings[home_team] - new_ratings[away_team]

            # Calculate rating change based on prediction error
            prediction_error = point_diff - expected_margin
            rating_change = learning_rate * prediction_error

            # Update ratings
            new_ratings[home_team] += rating_change
            new_ratings[away_team] -= rating_change

        # Update the instance ratings
        self.team_ratings = new_ratings

        # Recalculate final rating differences
        df['power_rating_diff'] = df.apply(
            lambda row: self.team_ratings[row['team_home']] - self.team_ratings[row['team_away']],
            axis=1
        )

        # Validate power ratings
        self._validate_power_ratings(df)

        return df

    def _validate_power_ratings(self, df: pd.DataFrame) -> None:
        """
        Validate power ratings and print warnings for potential issues
        """
        # Check if ratings are being updated
        if len(set(df['power_rating_diff'])) <= 1:
            print("Warning: Power rating differences show no variation")

        # Check rating distribution
        rating_values = list(self.team_ratings.values())
        rating_std = np.std(rating_values)
        if rating_std < 1 or rating_std > 20:
            print(f"Warning: Power rating standard deviation ({rating_std:.2f}) seems unusual")

        # Check correlation with point differential
        corr = df['power_rating_diff'].corr(df['point_differential'])
        if abs(corr) < 0.1:
            print(f"Warning: Low correlation between power ratings and point differential ({corr:.2f})")