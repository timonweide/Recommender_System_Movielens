# Recommender_System_Movielens

A movie recommendation system built using the popular MovieLens dataset. This project demonstrates various collaborative filtering and content-based filtering techniques implemented in Python.

## Features

- Movie recommendation using collaborative filtering (user-based and item-based)
- Content-based recommendation
- Evaluation metrics (RMSE, MAE, precision, recall, etc.)
- Data visualization and exploratory analysis
- Modular and extensible code structure

## Dataset

This project uses the [MovieLens dataset](https://grouplens.org/datasets/movielens/), a widely used benchmark dataset for movie recommendation research.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/timonweide/Recommender_System_Movielens.git
   cd Recommender_System_Movielens
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the MovieLens dataset (if not already included) and place it in the appropriate directory.

## Usage

1. Prepare the data:
   - Place the MovieLens CSV files in the designated data directory.

2. Run the streamlit app:
   ```bash
   streamlit run rec_sys_app.py
   ```

3. Choose algorithms and users inside the dashboard.

## Project Structure

```
├── data/                # Dataset files
├── rec_sys_algos.py     # All functions and algorithms needed for recommending and evaluating
├── rec_sys_app.py       # Streamlit App to provide interactive dashboard
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Algorithms Implemented

- Non-personalized filtering
- User-based collaborative filtering
- Item-based collaborative filtering
- Content-based filtering
- Hybrid approaches

## Evaluation

The system includes functions for evaluating recommendation performance using metrics such as RMSE, MAE, and top-N recommendation accuracy.

Metric leaders of offline evaluation:

Precision@10 - User‑CF (Cosine) - 0.1214

Recall@10 - User‑CF (Cosine) - 0.1860

MRR@10 - Hybrid CF + Weighted - 0.4355

RMSE - Avg Rating Weighted - 0.3744

## Acknowledgements

- [MovieLens](https://grouplens.org/datasets/movielens/)
- GroupLens Research
