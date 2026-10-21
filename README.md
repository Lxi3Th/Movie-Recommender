# Movie Recommender

Content-based movie recommender using TF-IDF and k-nearest neighbors (cosine distance).

You type a genre and keywords, and it prints the closest matching movies from the TMDB dataset.

## How it works

1. Loads the TMDB dataset and drops rows missing `genres`, `keywords`, or `title`
2. Combines each movie's genres and keywords into one text string
3. Converts every movie's text to a TF-IDF vector (`fit_transform`)
4. Converts your input to a vector with the same vocabulary (`transform`)
5. Finds the 3 nearest movies with `NearestNeighbors` (cosine metric)
6. Prints the titles, skipping duplicates

## Setup

```
pip install pandas scikit-learn
```

1. Download the dataset: https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates
2. Put `TMDB_all_movies.csv` in the same folder as `Knn.py`
3. Run:

```
python Knn.py
```

## Example

```
What Genre you like to watch : [YOUR INPUT]
What keyword(S) are you looking for: [YOUR INPUT]

Movie Recommendation 1 id number (...): ...
Movie Recommendation 2 id number (...): ...
Movie Recommendation 3 id number (...): ...
```

## Tech

- Python
- pandas
- scikit-learn (`TfidfVectorizer`, `NearestNeighbors`, `LabelEncoder`)

## Notes

The dataset is not included because of its size. Download it from the link above.
