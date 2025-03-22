import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the dataset (add the correct path to your dataset)
df = pd.read_csv('final_synthetic_dementia_user_data.csv')

# Feature Extraction
df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert timestamps
df['hour_of_day'] = df['timestamp'].dt.hour  # Hour of the day feature
df['day_of_week'] = df['timestamp'].dt.dayofweek  # Day of the week feature
df['engagement_duration'] = df['engagement_duration'].apply(lambda x: round(x, 2))  # Ensure float precision

# Scale numerical features for ML models
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['engagement_duration']])

# Convert scaled features into a DataFrame for easier inspection
scaled_df = pd.DataFrame(scaled_features, columns=['engagement_duration'])

# Apply K-means clustering to segment the users
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_df)  # Assign cluster labels to the dataset

# Check the clustering performance using silhouette score (higher values indicate better clusters)
silhouette_avg = silhouette_score(scaled_df, df['cluster'])

# Streamlit Display
st.title("Dementia User Engagement Analysis")
st.write("### Silhouette Score of Clustering:", silhouette_avg)

# Function to assign personalized notification schedules
def assign_personalized_schedule(row):
    notification_times = []
    
    # Handle low engagement users with multiple notifications during their preferred time
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
    
    # Handle high and moderate engagement users with a single daily notification
    else:
        if row['preferred_time'] == 'morning':
            return ['8:00 AM'], 'Once a day'
        elif row['preferred_time'] == 'afternoon':
            return ['1:00 PM'], 'Once a day'
        else:  # Evening preference
            return ['5:00 PM'], 'Once a day'

# Apply the function to assign personalized schedules
df['notification_schedule'] = df.apply(assign_personalized_schedule, axis=1)

# Display a preview of the DataFrame with the assigned schedules
st.write("### User Notification Schedules")
st.dataframe(df[['user_id', 'preferred_time', 'notification_schedule']].head())

# Optional: User interaction to view details of any specific user
user_id = st.selectbox("Select a User to View Details", df['user_id'].unique())
user_details = df[df['user_id'] == user_id]
st.write(f"### Details of User {user_id}")
st.write(user_details[['user_id', 'preferred_time', 'engagement_duration', 'cluster', 'notification_schedule']])

