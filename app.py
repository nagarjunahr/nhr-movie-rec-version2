from rapidfuzz import process

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
from surprise import SVD, Dataset, Reader

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


# Load the trained model and movie title mapping
model = joblib.load('svd_model.pkl')
movie_titles = joblib.load('movie_titles.pkl')

# Re-create the dataset for Surprise
# (needed to get trainset and inner/outer id mapping)
ratings = pd.read_csv('u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
reader = Reader(rating_scale=(1, 5))
data_for_surprise = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
trainset = data_for_surprise.build_full_trainset()

# Build a reverse lookup: title → movie_id
title_to_id = {v: k for k, v in movie_titles.items()}

# Recommendation logic using Surprise predictions

def get_recommendations(movie_title):
    # Use fuzzy matching to find closest title
    match, score, _ = process.extractOne(movie_title, title_to_id.keys())

    if score < 60:  # threshold, adjust if needed
        return [f"No close match found for '{movie_title}'. Try again?"]

    movie_id = title_to_id[match]
    all_users = trainset.all_users()

    # Predict this movie's rating for all users, find top users
    top_users = sorted(
        [(uid, model.predict(trainset.to_raw_uid(uid), movie_id).est) for uid in all_users],
        key=lambda x: x[1],
        reverse=True
    )[:10]

    # Recommend other movies to those users
    all_movie_ids = trainset.all_items()
    recommendations = {}

    for uid, _ in top_users:
        for mid in all_movie_ids:
            raw_uid = trainset.to_raw_uid(uid)
            raw_mid = trainset.to_raw_iid(mid)
            if raw_mid == movie_id:
                continue
            est = model.predict(raw_uid, raw_mid).est
            recommendations[raw_mid] = recommendations.get(raw_mid, 0) + est

    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:5]
    return [movie_titles[int(mid)] for mid, _ in sorted_recs]



@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    movie = data.get('movie')
    recommendations = get_recommendations(movie)
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)
