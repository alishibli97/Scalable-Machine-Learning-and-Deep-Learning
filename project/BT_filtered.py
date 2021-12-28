import pandas as pd
bt_history = pd.read_csv("NetflixViewingHistory.csv")
movies = pd.read_csv("movie_titles.csv")
movies.head(5)