import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load your dataset
df = pd.read_csv('final_synthetic_dementia_user_data.csv')

# Handle invalid timestamps and convert
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df['hour_of_day'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['engagement_duration'] = df['engagement_duration'].apply(lambda x: round(x, 2))

# Scaling numerical features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['engagement_duration', 'click_duration']])
scaled_df = pd.DataFrame(scaled_features, columns=['engagement_duration', 'click_duration'])

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_df)

# Function to assign personalized notification schedules
def assign_personalized_schedule(row):
    notification_times = []
    
    if row['engagement_duration'] < 30:  # Low engagement
        if row['preferred_time'] == 'morning':
            for i in range(7, 10):
                for minute in [0, 20, 40]:  # Every 20 minutes in the morning (7-10 AM)
                    notification_times.append(f'{i}:{minute:02d} AM')
        elif row['preferred_time'] == 'afternoon':
            for i in range(1, 4):
                for minute in [0, 20, 40]:  # Every 20 minutes in the afternoon (1-4 PM)
                    notification_times.append(f'{i}:{minute:02d} PM')
        else:  # Evening preference
            for i in range(5, 8):
                for minute in [0, 20, 40]:  # Every 20 minutes in the evening (5-8 PM)
                    notification_times.append(f'{i}:{minute:02d} PM')
        
        return notification_times, "Multiple times a day"
    
    else:  # High and moderate engagement users
        if row['preferred_time'] == 'morning':
            return ['8:00 AM'], 'Once a day'
        elif row['preferred_time'] == 'afternoon':
            return ['1:00 PM'], 'Once a day'
        else:  # Evening preference
            return ['5:00 PM'], 'Once a day'

# Streamlit UI for inputs
st.title("Dementia User Data Prediction")

# Input fields for engagement_duration, preferred_time, and click_duration
engagement_duration = st.number_input("Enter Engagement Duration (minutes)", min_value=0, max_value=200, value=30)
preferred_time = st.selectbox("Preferred Time for Notifications", options=["morning", "afternoon", "evening"])
click_duration = st.number_input("Enter Click Duration (seconds)", min_value=0, max_value=1000, value=20)

# Submit button to show predictions
if st.button('Submit'):
    # Create a new row based on user inputs for prediction
    new_data = pd.DataFrame({
        'engagement_duration': [engagement_duration],
        'preferred_time': [preferred_time],
        'click_duration': [click_duration],
        'timestamp': [pd.Timestamp.now()]  # Using current timestamp as placeholder
    })

    # Process the new data (same feature extraction and scaling as done for the original dataset)
    new_data['hour_of_day'] = new_data['timestamp'].dt.hour
    new_data['day_of_week'] = new_data['timestamp'].dt.dayofweek
    new_data['engagement_duration'] = new_data['engagement_duration'].apply(lambda x: round(x, 2))

    # Scaling features for the new input data
    scaled_new_data = scaler.transform(new_data[['engagement_duration', 'click_duration']])
    scaled_new_data_df = pd.DataFrame(scaled_new_data, columns=['engagement_duration', 'click_duration'])

    # Predict the cluster for the new data
    new_data['cluster'] = kmeans.predict(scaled_new_data_df)

    # Assign personalized notification schedule (now works correctly)
    new_data['notification_schedule'] = new_data.apply(lambda row: assign_personalized_schedule(row), axis=1)

    # Display results
    st.write(f"Cluster: {new_data['cluster'].values[0]}")
    st.write(f"Notification Schedule: {new_data['notification_schedule'].values[0][0]}")
    st.write(f"Notification Frequency: {new_data['notification_schedule'].values[0][1]}")
