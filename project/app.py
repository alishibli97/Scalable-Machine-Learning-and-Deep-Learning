from flask import Flask, render_template, request, flash
from suggestion_model import SuggestionModel

app = Flask(__name__)

model = SuggestionModel()
movies = None # model.sample_movies(k=10)

num_movies = 5
num_genres = 5

# movies = ["GOT","Vikings","Playing Football","Twist","To wok"]

@app.route("/",methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        global movies
        checked_movies = []
        for movie in movies.title.tolist():
            if request.form.get(movie.split()[0]) != None:
                checked_movies.append(movie)
                print("You liked the movie:",movie)
        if len(checked_movies)>0:
            model.update_user(checked_movies)
            movies=model.recommend_movies(k=num_movies)
            movies_abstracts = list(zip(movies.title.tolist(),movies.description.tolist()))
            genres = [k for k in sorted(model.user.user_genres.items(), key=lambda item: item[1], reverse=True)][:num_genres]
            return render_template("index.html",movies_abstracts=movies_abstracts,genres=genres)
        else:
            movies = model.sample_movies(k=num_movies)
            movies_abstracts = list(zip(movies.title.tolist(),movies.description.tolist()))
            genres = [k for k in sorted(model.user.user_genres.items(), key=lambda item: item[1], reverse=True)][:num_genres]
            return render_template("index.html",movies_abstracts=movies_abstracts,genres=genres)
    else:
        movies = model.sample_movies(k=num_movies)
        movies_abstracts = list(zip(movies.title.tolist(),movies.description.tolist()))
        genres = [k for k in sorted(model.user.user_genres.items(), key=lambda item: item[1], reverse=True)][:num_genres]
        return render_template("index.html",movies_abstracts=movies_abstracts,genres=genres)