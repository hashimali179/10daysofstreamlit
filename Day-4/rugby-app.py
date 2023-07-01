import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title('NFL Football Stats (Rushing) Explorer')

st.markdown("""
This app performs simple webscraping of NFL Football player stats data (focusing on Rushing)!
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Data source:** [pro-football-reference.com](https://www.pro-football-reference.com/).
""")

st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1990,2023))))

# Web scraping of NFL player stats
# https://www.pro-football-reference.com/years/2022/scoring.htm

@st.cache_resource
def load_data(year):
    url = "https://www.pro-football-reference.com/years/" + str(year) + "/scoring.htm"
    html = pd.read_html(url, header = 0)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index) # Deletes repeating headers in content
    raw = raw.fillna(0)
    playerstats = raw.drop(['Rk'], axis=1)
    return playerstats
playerstats = load_data(selected_year)

# Sidebar - Team selection
sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)

# Sidebar - Position selection
unique_pos = ['RB','QB','WR','FB','TE', 'K', 'DB']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

# Filtering data
df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

st.header('Display Player Stats of Selected Team(s)')
st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
st.dataframe(df_selected_team)

# Download NBA player stats data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

if st.button("Show Charts"):
    df_selected_team.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')

    # Calculate the average age and average points per team
    avg_age_by_team = df.groupby('Tm')['Age'].mean().sort_values(ascending=False).head(10)
    avg_points_by_team = df.groupby('Tm')['Pts'].mean().sort_values(ascending=False).head(10)
    avg_age_by_position = df.groupby('Pos')['Age'].mean().sort_values(ascending=False).head(5)
    avg_points_by_position = df.groupby('Pos')['Pts'].mean().sort_values(ascending=False).head(5)

    # Create subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    # Plotting the average age/team bar chart
    ax1 = axes[0][0]
    avg_age_by_team.plot(kind='bar', ax=ax1)
    ax1.set_xlabel('Team')
    ax1.set_ylabel('Average Age')
    ax1.set_title('Average Age of Players by Team (Top 10)')

    # Plotting the average points/team bar chart
    ax2 = axes[0][1]
    avg_points_by_team.plot(kind='bar', ax=ax2)
    ax2.set_xlabel('Team')
    ax2.set_ylabel('Average Points')
    ax2.set_title('Average Points Scored by Team (Top 10)')

    # Plotting the average age/position bar chart
    ax3 = axes[1][0]
    avg_age_by_position.plot(kind='bar', ax=ax3)
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Average Age')
    ax3.set_title('Average Age by Position Played (Top 5)')

    # Plotting the average points/position bar chart
    ax4 = axes[1][1]
    avg_points_by_position.plot(kind='bar', ax=ax4)
    ax4.set_xlabel('Position')
    ax4.set_ylabel('Average Points')
    ax4.set_title('Average Points Scored by Position Played (Top 5)')

    # Adjust the spacing between subplots
    plt.tight_layout()
    # Display the plot
    st.pyplot(fig)