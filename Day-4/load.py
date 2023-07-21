import pandas as pd
def load_data(year):
    url = "https://www.pro-football-reference.com/years/" + str(year) + "/scoring.htm"
    html = pd.read_html(url, header = 0)
    df = html[0]
    #raw = df.drop(df[df.Age == 'Age'].index) # Deletes repeating headers in content
    raw = df.fillna(0)
    #playerstats = raw.drop(['Rk'], axis=1)
    return raw
raw = load_data(2021)
position = raw['Pos'].value_counts()
print(position)