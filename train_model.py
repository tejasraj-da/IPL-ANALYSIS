import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib # Used for saving the model and encoder

print("Starting the training process...")

# --- 1. Load and Clean Data ---
try:
    ipl_matches_data = pd.read_csv('ipl_matches_data.csv')
    ball_by_ball_data = pd.read_csv('ball_by_ball_data.csv')
except FileNotFoundError:
    print("Error: Make sure 'ipl_matches_data.csv' and 'ball_by_ball_data.csv' are in the same directory.")
    exit()

# Standardize team names
team_name_mapping = {
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad'
}
for col in ['team1', 'team2', 'match_winner', 'toss_winner']:
    ipl_matches_data[col] = ipl_matches_data[col].replace(team_name_mapping)

data = pd.merge(left=ipl_matches_data, right=ball_by_ball_data, on='match_id', how='right')


# --- 2. Player Role Classification ---
player_batting_stats = data.groupby('batter').agg(total_runs=('batter_runs', 'sum')).reset_index()
wicket_kinds = ['caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']
player_bowling_stats = data[data['wicket_kind'].isin(wicket_kinds)].groupby('bowler').agg(total_wickets=('is_wicket', 'sum')).reset_index()
player_stats = pd.merge(player_batting_stats, player_bowling_stats, left_on='batter', right_on='bowler', how='outer').fillna(0)
player_stats.rename(columns={'batter': 'player_name'}, inplace=True)

RUNS_THRESHOLD = 1000
WICKETS_THRESHOLD = 50
def classify_player(row):
    is_batsman = row['total_runs'] >= RUNS_THRESHOLD
    is_bowler = row['total_wickets'] >= WICKETS_THRESHOLD
    if is_batsman and is_bowler: return 'All-Rounder'
    elif is_batsman: return 'Batsman'
    elif is_bowler: return 'Bowler'
    else: return 'Batsman' if row['total_runs'] > row['total_wickets'] * 20 else 'Bowler'
player_stats['player_role'] = player_stats.apply(classify_player, axis=1)


# --- 3. Feature Engineering for ML ---
players_in_match = data.groupby(['match_id', 'team_batting'])['batter'].unique().apply(list).reset_index()
players_in_match.rename(columns={'batter': 'player_name_list'}, inplace=True)
team_composition = players_in_match.explode('player_name_list').rename(columns={'player_name_list': 'player_name'})
player_roles = player_stats[['player_name', 'player_role']]
team_composition = pd.merge(team_composition, player_roles, on='player_name', how='left').fillna('Bowler')
composition_counts = team_composition.groupby(['match_id', 'team_batting', 'player_role']).size().unstack(fill_value=0).reset_index()

team1_composition = composition_counts.rename(columns={'team_batting': 'team1', 'Batsman': 'team1_batsmen', 'All-Rounder': 'team1_allrounders', 'Bowler': 'team1_bowlers'})
team2_composition = composition_counts.rename(columns={'team_batting': 'team2', 'Batsman': 'team2_batsmen', 'All-Rounder': 'team2_allrounders', 'Bowler': 'team2_bowlers'})

ml_df = ipl_matches_data[['match_id', 'city', 'team1', 'team2', 'toss_winner', 'toss_decision', 'match_winner']].copy()
ml_df = pd.merge(ml_df, team1_composition, on=['match_id', 'team1'], how='left')
ml_df = pd.merge(ml_df, team2_composition, on=['match_id', 'team2'], how='left')
ml_df.dropna(inplace=True)

all_teams = pd.concat([ml_df['team1'], ml_df['team2'], ml_df['toss_winner'], ml_df['match_winner']]).unique()
team_encoder = LabelEncoder().fit(all_teams)

ml_df['team1_encoded'] = team_encoder.transform(ml_df['team1'])
ml_df['team2_encoded'] = team_encoder.transform(ml_df['team2'])
ml_df['toss_winner_encoded'] = team_encoder.transform(ml_df['toss_winner'])
ml_df['match_winner_encoded'] = team_encoder.transform(ml_df['match_winner'])

ml_df = pd.get_dummies(ml_df, columns=['city', 'toss_decision'], drop_first=True)


# --- 4. Model Training ---
features = [col for col in ml_df.columns if col not in ['match_id', 'team1', 'team2', 'toss_winner', 'match_winner', 'match_winner_encoded']]
X = ml_df[features]
y = ml_df['match_winner_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)


# --- 5. Save the Model and Supporting Files ---
joblib.dump(model, 'ipl_model.joblib')
joblib.dump(team_encoder, 'team_encoder.joblib')
joblib.dump(X.columns.tolist(), 'feature_columns.joblib')

print("Model, encoder, and feature columns have been saved successfully.")

