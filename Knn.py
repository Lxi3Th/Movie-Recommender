import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer


db = pd.read_csv('TMDB_all_movies.csv')
db = db.dropna(subset=['genres', 'keywords', 'title']).reset_index(drop=True)


X = db[['genres', 'keywords']]
Y = db['title']


title_encoder = LabelEncoder()
db['title'] = title_encoder.fit_transform(db['title'])


qn = input('What Genre you like to watch : ')
key = input('What keyword(S) are you looking for: ')


Nvar = X['genres'] + " " + X['keywords']
tfdv = TfidfVectorizer()
x = tfdv.fit_transform(Nvar)


model = NearestNeighbors(n_neighbors=2 + 1, metric='cosine')
model.fit(x)


query = tfdv.transform([qn + " " + key])
distance, indices = model.kneighbors(query)
MovieSlicer = indices


moviesSeen = []
count = 0

for movieNames in MovieSlicer[0]:
    title_id = db['title'].iloc[movieNames]
    showTitle = title_encoder.inverse_transform([title_id])

    if showTitle[0] not in moviesSeen:
        moviesSeen.append(showTitle[0])
        count += 1
        print(f'Movie Recommendation {count} id number ({movieNames}): {showTitle[0]}')