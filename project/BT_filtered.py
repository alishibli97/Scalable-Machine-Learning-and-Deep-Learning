import pandas as pd

def remove_seasons(row):
    if ": Säsong " in row['Title']:
       return(row['Title'].split(": Säsong ")[0])
    return row["Title"]

bt_history = pd.read_csv('NetflixViewingHistory.csv', sep=';', delimiter=None, header='infer', names=None, index_col=None)
movies = pd.read_csv("movie_titles.csv", sep=';', delimiter=None, header='infer', names=None, index_col=None)


bt_history['Title'] = bt_history.apply (lambda row: remove_seasons(row), axis=1)


updated_history = bt_history.merge(movies.drop_duplicates(), on = ["Title"], 
                   how='left', indicator=True)

updated_history = updated_history[updated_history["_merge"]=="both"]
updated_history = updated_history[["Title","Date"]]

print(bt_history.head(5))
print(updated_history.head(5))