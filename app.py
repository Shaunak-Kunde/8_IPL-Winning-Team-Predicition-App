import streamlit as st
import joblib
import os
import pandas as pd

# IPL Teams
teams = [
    'Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
    'Delhi Daredevils', 'Sunrisers Hyderabad'
]

# IPL Venues
venues = [
    'M Chinnaswamy Stadium', 'Punjab Cricket Association Stadium, Mohali',
    'Feroz Shah Kotla', 'Wankhede Stadium', 'Eden Gardens',
    'Sawai Mansingh Stadium', 'Rajiv Gandhi International Stadium, Uppal',
    'MA Chidambaram Stadium, Chepauk', 'Dr DY Patil Sports Academy', 'Newlands',
    "St George's Park", 'Kingsmead', 'SuperSport Park', 'Buffalo Park',
    'New Wanderers Stadium', 'De Beers Diamond Oval', 'OUTsurance Oval',
    'Brabourne Stadium', 'Sardar Patel Stadium, Motera', 'Barabati Stadium',
    'Vidarbha Cricket Association Stadium, Jamtha',
    'Himachal Pradesh Cricket Association Stadium', 'Nehru Stadium',
    'Holkar Cricket Stadium',
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
    'Subrata Roy Sahara Stadium',
    'Shaheed Veer Narayan Singh International Stadium',
    'JSCA International Stadium Complex', 'Sheikh Zayed Stadium',
    'Sharjah Cricket Stadium', 'Dubai International Cricket Stadium',
    'Maharashtra Cricket Association Stadium',
    'Punjab Cricket Association IS Bindra Stadium, Mohali',
    'Saurashtra Cricket Association Stadium', 'Green Park'
]

# Load model and training columns
model_path = "forest_classifier_shaunak.joblib"
columns_path = "training_columns_shaunak.joblib"
pipe = None
training_columns = None

if os.path.exists(model_path):
    try:
        pipe = joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.warning(f"Model file not found: {model_path}")

if os.path.exists(columns_path):
    try:
        training_columns = joblib.load(columns_path)
    except Exception as e:
        st.error(f"Error loading training columns: {e}")
else:
    st.warning(f"Training columns file not found: {columns_path}")

# UI
st.title("🏏 IPL Win Predictor by Shaunak")

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the Bowling Team', sorted(teams))

venue_selected = st.selectbox('Select Venue', venues)

target = st.number_input('Target Score', min_value=1, max_value=300, step=1)
score = st.number_input('Current Score', min_value=0, max_value=300, step=1)
overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10, step=1)

if st.button("Predict Win Probability"):
    if pipe is None or training_columns is None:
        st.error("Model or training columns not loaded.")
    else:
        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets_remaining = 10 - wickets
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6 / balls_left) if balls_left > 0 else 0

        # Prepare input
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'venue': [venue_selected],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets_remaining': [wickets_remaining],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Encode & align with training columns
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=training_columns, fill_value=0)

        # Predict win probability using predict_proba()
        prediction = pipe.predict_proba(input_df)
        
        # Win probability is for class 1 (win)
        win_prob = prediction[0][1] * 100
        # Loss probability for the batting team is win probability for the bowling team
        loss_prob = prediction[0][0] * 100

        st.subheader("Prediction:")
        st.success(f"**{batting_team}** Win Probability: **{win_prob:.2f}%**")
        st.error(f"**{bowling_team}** Win Probability: **{loss_prob:.2f}%**")