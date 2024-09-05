from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

# Загружаем сохранённые данные и модели
netflix_data = pd.read_csv('netflix_data.csv')
cosine_sim = np.load('model/cosine_sim_matrix.npy')
with open('model/tfidf_vectorizer.pkl', 'rb') as f:
    tfid = pickle.load(f)

class FlixHub:
    def __init__(self, df, cosine_sim):
        self.df = df
        self.cosine_sim = cosine_sim

    def recommendation(self, title, total_result=5):
        idx = self.find_id(title)
        if idx == -1:
            return [], []
        self.df['similarity'] = self.cosine_sim[idx]
        sort_df = self.df.sort_values(by='similarity', ascending=False)[1:total_result+1]

        movies = sort_df['title'][sort_df['type'] == 'Movie']
        tv_shows = sort_df['title'][sort_df['type'] == 'TV Show']

        return list(movies), list(tv_shows)

    def find_id(self, name):
        for index, string in enumerate(self.df['title']):
            if name.lower() in string.lower():
                return index
        return -1

# Инициализируем Flask-приложение
app = Flask(__name__)

# Загружаем рекомендательную систему
flix_hub = FlixHub(netflix_data, cosine_sim)

# Маршрут для главной страницы
@app.route('/')
def index():
    return render_template('index.html')

# Маршрут для получения рекомендаций
@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie_name']
    movies, tv_shows = flix_hub.recommendation(movie_name, total_result=10)

    return render_template('index.html', movie_name=movie_name, movies=movies, tv_shows=tv_shows)

if __name__ == '__main__':
    app.run(debug=True)
