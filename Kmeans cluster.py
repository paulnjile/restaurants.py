from sklearn.cluster import KMeans

movies = [
    {"title": "Movie 1", "features": [9, 1]},  # lots of action, little sci fi
    {"title": "Movie 2", "features": [8, 2]},  # lots of action, some sci fi
    {"title": "Movie 3", "features": [3, 7]},  # some action, lots of sci fi
    {"title": "Movie 4", "features": [1, 9]},  # little action, lots of sci fi
    # ...
]

feature_vectors = [movie["features"] for movie in movies]

kmeans = KMeans(n_clusters=2).fit(feature_vectors)  # Corrected variable name

liked_movie = "Movie 1"  # Corrected variable name

liked_movie_features = next(movie["features"] for movie in movies if movie["title"] == liked_movie)

liked_movie_cluster = kmeans.predict([liked_movie_features])[0]

recommended_movie = next(movie for movie in movies if movie["title"] != liked_movie and kmeans.predict([movie["features"]])[0] == liked_movie_cluster)

print("Because you liked Movie 1, we recommend: " + recommended_movie["title"])  # Corrected variable name
