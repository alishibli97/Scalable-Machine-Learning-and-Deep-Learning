import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial import cKDTree

class SuggestionModel:
    def __init__(self):
        self.df = pd.read_csv("netflix_titles.csv")
        self.df.dropna(inplace=True)
        self.df.reset_index(inplace=True,drop=True)
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = np.load("embeddings.npy")
        self.user = User(self.get_genres())
        self.initialize_user("filtered_BT_history.csv")
        self.suggested_movies = []
    
    def get_genres(self):
        genres = set([x.replace(',', '') for k in self.df.listed_in.tolist() for x in k.split()])
        genres.remove("&")
        return genres

    def initialize_user(self,path):
        print("Initializing user info")
        df = pd.read_csv(path)
        watched_movies = set(df.Title.tolist())
        self.update_user(watched_movies)

    def sample_movies(self,k=10):
        print(f"Sampling {k} new movies out of {len(self.df)}")
        sampled = self.df.sample(k)
        return sampled

    def update_user(self,checked_movies):
        print("Updating user vectors")
        subset = self.df.loc[self.df['title'].isin(checked_movies)]
        data = []
        for descr in subset.description:
            emb = self.sbert.encode(descr)
            data.append(emb)
        data = np.array(data)
        genres = set([x.replace(',', '') for k in subset.listed_in.tolist() for x in k.split()])
        if "&" in genres: genres.remove("&")
        self.user.update_user_info(data,genres,checked_movies)

    def recommend_movies(self,k=10):
        print(f"Recommending {k} new movies out of {len(self.df)}")
        indexes = cKDTree(self.embeddings).query(self.user.user_vector, k=k)[1]
        print(indexes)
        recommended = self.df.iloc[indexes]
        return recommended

class User:
    def __init__(self,genres):
        self.user_vector = np.zeros(384)
        self.sum_vector = np.zeros(384)
        self.total_number_of_liked_movies = 0
        self.user_genres = {genre:0 for genre in genres}
        self.checked_movies = []

    def update_user_info(self,new_data,new_genres,checked_movies):
        self.sum_vector += new_data.sum(0)
        self.total_number_of_liked_movies += len(new_data)
        self.user_vector = self.sum_vector / self.total_number_of_liked_movies

        for genre in new_genres:
            self.user_genres[genre]+=1

        self.checked_movies.extend(checked_movies)

        print(len(set(self.checked_movies)))