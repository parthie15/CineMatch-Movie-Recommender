# 🎬 CineMatch — Movie Recommendation System

CineMatch is a movie discovery application built with Python and Streamlit. It recommends movies similar to a selected film using content-based filtering.

## Features

- Search and select movies
- Recommend similar movies using TF-IDF and cosine similarity
- Display movie posters when available
- Show average ratings and similarity scores
- Interactive Streamlit interface

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- TF-IDF vectorization
- Cosine similarity

## Dataset Setup

CineMatch uses the **MovieLens `ml-latest-small` dataset** from GroupLens Research.

### 1. Download the dataset

Download `ml-latest-small.zip` from the official [MovieLens dataset page](https://grouplens.org/datasets/movielens/latest/).

Extract the ZIP file.

### 2. Create the data folder

Inside the CineMatch project folder, create a folder named `data`.

Copy these four files from the extracted `ml-latest-small` folder into `data/`:

- `movies.csv`
- `ratings.csv`
- `tags.csv`
- `links.csv`

Your project structure should look like this:

```text
CineMatch/
├── app.py
├── requirements.txt
├── README.md
└── data/
    ├── movies.csv
    ├── ratings.csv
    ├── tags.csv
    └── links.csv
```

### 3. Optional movie posters

The application can display posters using `movie_posters.csv`.

If you have a compatible poster mapping file, place it in the `data/` folder. Make sure you have permission to use and redistribute the poster mapping and comply with the image provider's terms.

If the poster file is unavailable, update the application to handle its absence gracefully.

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/parthie15/CineMatch-Movie-Recommender.git
cd CineMatch-Movie-Recommender
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Follow the Dataset Setup instructions above before running the app.

### 4. Launch CineMatch

```bash
streamlit run app.py
```

The application will open in your browser.

## How It Works

1. Loads movie metadata, ratings, and user-generated tags.
2. Combines movie genres and tags into text descriptions.
3. Converts descriptions into numerical features using TF-IDF.
4. Calculates cosine similarity between movies.
5. Displays similar movies with available ratings and posters.

## Limitations

- Recommendations are based on movie content, not individual user viewing histories.
- Some external poster links may be unavailable.
- Similarity scores measure content similarity, not a guaranteed prediction of user preference.

## Dataset Acknowledgment

MovieLens is provided by GroupLens Research at the University of Minnesota. Please review the dataset's included README and usage terms before using or redistributing the data.

## Future Improvements

- Add collaborative filtering
- Improve recommendation ranking
- Add genre and rating filters
- Deploy a live demo
