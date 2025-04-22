# app.py
from flask import Flask, render_template, request, url_for
from functools import lru_cache
import os
import random
import pandas as pd
import requests
import json
import google.generativeai as genai
from datetime import datetime

app = Flask(__name__, static_url_path='/static')
DATA_FILE = "IPLBAT4.xlsx"
CACHE_SIZE = 128

# API Keys
CRICAPI_KEY = "f8df0022-e2d7-495e-9882-19afefb83c7c"
GEMINI_API_KEY = "AIzaSyCIje_WZn3jJOhM2yE4rHsWJiMDNImbykk"
GEMINI_MODEL_NAME = "gemini-1.5-pro"
CRICAPI_URL = f"https://api.cricapi.com/v1/currentMatches?apikey={CRICAPI_KEY}&offset=0&limit=10"

# Initialize Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL_NAME)

# Load team data for team comparison
try:
    team_df = pd.read_excel("ipl.xlsx", sheet_name="Sheet1")
    team_df = team_df[['Team', 'Matches Played', 'Wins']].dropna()
    team_df['Win Rate'] = team_df['Wins'] / team_df['Matches Played']
except FileNotFoundError:
    print("❌ ERROR: 'ipl.xlsx' not found. Team comparison functionality will be disabled.")
    team_df = None

# Load player data with caching
@lru_cache(maxsize=CACHE_SIZE)
def load_data():
    try:
        file_path = os.path.join(os.path.dirname(__file__), DATA_FILE)
        df = pd.read_excel(file_path, sheet_name="Batsmen", skiprows=2, header=0)
        df.columns = [
            "Player", "Role", "Season_Performance", "Team", "Match_5_vs_MI", 
            "Match_10_vs_RCB", "Match_14_vs_PBKS", "Match_18_vs_LSG", 
            "Match_23_vs_MI", "Match_28_vs_SRH", "Match_31_vs_CSK", 
            "Match_32_vs_RCB", "Unused", "Wickets"
        ]

        df = df.dropna(subset=["Player"]).drop(columns=["Unused"])
        df[["Player", "Role", "Team"]] = df[["Player", "Role", "Team"]].fillna("").apply(lambda x: x.str.strip())

        match_cols = [col for col in df.columns if col.startswith("Match")]
        tidy_df = pd.melt(
            df, id_vars=["Player", "Role", "Team", "Wickets"], 
            value_vars=match_cols, var_name="Match", value_name="Performance"
        )

        tidy_df["Opponent"] = tidy_df["Match"].str.extract(r'vs_([A-Z]+)')[0]
        tidy_df["Performance"] = tidy_df["Performance"].astype(str).str.replace('*', '').apply(pd.to_numeric, errors='coerce').fillna(0)
        tidy_df["Wickets"] = pd.to_numeric(tidy_df["Wickets"], errors='coerce').fillna(0)

        metrics = tidy_df.apply(
            lambda row: {
                'Runs': float(row["Performance"]) if "Bowler" not in row["Role"] else 0, 
                'Wickets': float(row["Wickets"]) if "Bowler" in row["Role"] else 0
            }, 
            axis=1, result_type='expand'
        )
        return pd.concat([tidy_df, metrics], axis=1)
    except Exception as e:
        raise RuntimeError(f"Data loading failed: {str(e)}")

def fetch_matches():
    try:
        print("Fetching Live and Upcoming Matches from Gemini API...")

        prompt_text = """
        Provide only JSON output for the latest live and upcoming cricket matches.
        The JSON format must be:
        {
            "matches": [
                {
                    "name": "Match Name",
                    "teams": ["Team 1", "Team 2"],
                    "venue": "Venue Name",
                    "status": "Live / Upcoming",
                    "scores": [
                        {"team": "Team 1", "runs": 120, "wickets": 3, "overs": 15.2},
                        {"team": "Team 2", "runs": 0, "wickets": 0, "overs": 0}
                    ]
                }
            ]
        }
        **Return only JSON. Do not add any extra text, explanation, or markdown formatting.**
        """

        gemini_response = model.generate_content(prompt_text)
        gemini_text = gemini_response.text.strip()

        print("Raw Gemini Response:\n", gemini_text)

        try:
            if gemini_text.startswith("```json"):
                gemini_text = gemini_text.replace("```json", "").replace("```", "").strip()
            gemini_data = json.loads(gemini_text)
        except json.JSONDecodeError:
            print("Error parsing Gemini response JSON.")
            gemini_data = {"matches": []}

        print("Fetching Completed Matches from CricAPI...")
        response_cricapi = requests.get(CRICAPI_URL, timeout=10)
        print(f"CricAPI Status Code: {response_cricapi.status_code}")

        cricapi_data = response_cricapi.json() if response_cricapi.status_code == 200 else None

        return {"gemini": gemini_data, "cricapi": cricapi_data}
    except Exception as e:
        print(f"Error fetching matches: {e}")
        return None

def process_matches(data):
    print("Processing matches...")
    if not data:
        print("No data received")
        return {'live': [], 'upcoming': [], 'completed': []}

    matches = {'live': [], 'upcoming': [], 'completed': []}

    if data.get('gemini'):
        gemini_matches = data['gemini'].get("matches", [])
        for match in gemini_matches:
            status = match.get("status", "").lower()
            processed_match = {
                "name": match.get("name", "Match"),
                "teams": match.get("teams", []),
                "venue": match.get("venue", "Venue not available"),
                "scores": match.get("scores", []),
                "status": match.get("status", "Unknown Status")
            }

            if "live" in status or "in progress" in status:
                matches['live'].append(processed_match)
            elif "upcoming" in status or "scheduled" in status:
                matches['upcoming'].append(processed_match)

    if data.get('cricapi') and 'data' in data['cricapi']:
        for match in data['cricapi']['data']:
            status = match.get('status', '').lower()
            if any(x in status for x in ['completed', 'won', 'result']):
                matches['completed'].append({
                    'id': match.get('id', ''),
                    'name': match.get('name', 'Match'),
                    'status': match.get('status', 'Status not available'),
                    'date': match.get('date', ''),
                    'venue': match.get('venue', 'Venue not available'),
                    'teams': match.get('teams', []),
                    'scores': match.get('score', []),
                    'matchType': match.get('matchType', ''),
                    'series': match.get('series', '')
                })

    print(f"Found {len(matches['live'])} live matches and {len(matches['upcoming'])} upcoming matches")
    return matches

def predict_performance(df, player, next_opponent):
    try:
        player_data = df[df["Player"] == player]
        if player_data.empty:
            return "No data available for this player"

        role = str(player_data["Role"].iloc[0])
        if "Bowler" in role:
            prediction = random.randint(1, 6)
            if prediction <= 2:
                return "Likely to take 2 wickets (Prediction range: 2–3 wickets)"
            elif prediction <= 4:
                return "In good form for 3 wickets (Prediction range: 3–4 wickets)"
            return "On fire! Might take 4 wickets (Prediction range: 4–5 wickets)"
        else:
            player_data["Runs"] = pd.to_numeric(player_data["Runs"], errors="coerce").fillna(0)
            avg_runs = player_data.tail(5)["Runs"].mean()
            vs_opponent_avg = player_data[player_data["Opponent"] == next_opponent]["Runs"].mean() \
                if not player_data[player_data["Opponent"] == next_opponent].empty else avg_runs
            prediction = (0.6 * avg_runs) + (0.4 * vs_opponent_avg)
            low, high = int(max(0, round(prediction * 0.9))), int(round(prediction * 1.1))
            if low == high:
                high += 5 if low < 50 else 10
            return f"{low}–{high} runs"
    except Exception as e:
        return f"Prediction error: {str(e)}"

@app.route("/")
def home():
    return render_template("index2.html")

# @app.route("/index")
# def original_home():
#     return render_template("index.html")

@app.route("/player", methods=["GET", "POST"])
def player_prediction():
    try:
        df = load_data()
        form_data = {k: request.form.get(k, "") for k in ["team", "role", "player", "opponent"]}
        form_data["player"] = form_data["player"] or request.form.get("preserve_player", "")
        form_data["opponent"] = form_data["opponent"] or request.form.get("preserve_opponent", "")

        teams = sorted(df["Team"].dropna().unique())
        opponents = sorted(df["Opponent"].dropna().unique())
        team_players = df[df["Team"] == form_data["team"]] if form_data["team"] else df

        if form_data["role"] == "batsman":
            players = sorted(team_players[~team_players["Role"].str.contains("Bowler")]["Player"].unique())
        elif form_data["role"] == "bowler":
            players = sorted(team_players[team_players["Role"].str.contains("Bowler")]["Player"].unique())
        else:
            players = sorted(team_players["Player"].unique())

        prediction = predict_performance(df, form_data["player"], form_data["opponent"]) \
            if form_data["player"] and form_data["opponent"] else None

        return render_template("player.html", teams=teams, players=players, opponents=opponents, prediction=prediction, **form_data)
    except Exception as e:
        return render_template("error.html", error_message=str(e))

@app.route("/team", methods=["GET", "POST"])
def team_comparison():
    if team_df is None:
        return render_template("error.html", error_message="Team data not available")
    
    prediction = None
    selected_team1 = selected_team2 = None
    teams = team_df['Team'].tolist()

    if request.method == "POST":
        selected_team1 = request.form.get("team1")
        selected_team2 = request.form.get("team2")

        if selected_team1 and selected_team2 and selected_team1 != selected_team2:
            win_rate1 = team_df[team_df['Team'] == selected_team1]['Win Rate'].values[0]
            win_rate2 = team_df[team_df['Team'] == selected_team2]['Win Rate'].values[0]

            total = win_rate1 + win_rate2
            team1_percentage = round((win_rate1 / total) * 100, 2)
            team2_percentage = round((win_rate2 / total) * 100, 2)

            if team1_percentage > team2_percentage:
                winner = selected_team1
            elif team2_percentage > team1_percentage:
                winner = selected_team2
            else:
                winner = "Tie"

            confidence = f"{abs(team1_percentage - team2_percentage):.2f}%"

            prediction = {
                "winner": winner,
                "team1_win_percentage": team1_percentage,
                "team2_win_percentage": team2_percentage,
                "confidence": confidence,
                "team1_win_rate": round(win_rate1 * 100, 2),
                "team2_win_rate": round(win_rate2 * 100, 2)
            }

    return render_template("team.html", 
                         teams=teams, 
                         prediction=prediction, 
                         team1=selected_team1, 
                         team2=selected_team2)

@app.route("/matches")
def show_matches():
    api_data = fetch_matches()
    matches = process_matches(api_data)
    return render_template('matches.html', 
                         matches=matches,
                         current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

@app.route("/analysis")
def analysis():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)