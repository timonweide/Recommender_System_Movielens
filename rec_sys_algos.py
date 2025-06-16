### All algorithms and helper functions for my Recommender System
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares


## Non-Personalized
def top_n_count(df, N=10, target_user=None):
    # By count of ratings
    counts = df['movieId'].value_counts()

    # Return top N movieIds and scores
    ids = counts.nlargest(N).index.tolist()
    scores = counts.nlargest(N).values.tolist()
    return ids, scores

def top_n_likes(df, N=10, threshold=4, target_user=None):
    # By count of likes (e.g. rating ≥4)
    likes = df[df['rating'] >= threshold]
    like_counts = likes['movieId'].value_counts()

    # Return top N movieIds and scores
    ids = like_counts.nlargest(N).index.tolist()
    scores = like_counts.nlargest(N).values.tolist()
    return ids, scores

def average_rating(df, N=10, target_user=None):
    # Calculate average ratings for each movie
    average_ratings = df.groupby('movieId')['rating'].mean()

    # Return top N movieIds and scores
    ids = average_ratings.nlargest(N).index.tolist()
    scores = average_ratings.nlargest(N).values.tolist()
    return ids, scores

def average_rating_normalized(df, N=10, target_user=None):
    # Compute user averages
    user_avg = df.groupby('userId')['rating'].mean().rename('user_avg')
    df = df.merge(user_avg, on='userId')

    # Compute predicted score S(u, i) for all (user, item) pairs
    predictions = []

    for _, row in df.iterrows():
        u = row['userId']
        i = row['movieId']
        user_avg_row = row['user_avg']

        # Get all ratings for item i
        item_ratings = df[df['movieId'] == i]

        # Check for cold start problem
        if item_ratings.empty:
            pred = user_avg_row
        else:
            deviation_sum = (item_ratings['rating'] - item_ratings['user_avg']).sum()
            normalized_score = deviation_sum / len(item_ratings)
            pred = user_avg_row + normalized_score

        predictions.append((i, pred))

    # Create a DataFrame for predictions
    pred_df = pd.DataFrame(predictions, columns=['movieId', 'predicted'])

    # Calculate normalized average ratings for each movie
    average_ratings_normalized = pred_df.groupby('movieId')['predicted'].mean()

    # Return top N movieIds and scores
    ids = average_ratings_normalized.nlargest(N).index.tolist()
    scores = average_ratings_normalized.nlargest(N).values.tolist()
    return ids, scores

def average_rating_weighted(df, N=10, m=10, target_user=None):
    # Compute average rating for each movie (U(j)) and number of votes (v)
    average_ratings_weighted = df.groupby('movieId')['rating'].agg(['mean', 'count']).rename(columns={'mean': 'U', 'count': 'v'})

    # Compute overall mean rating across all movies (C)
    C = df['rating'].mean()

    # Compute WR(j) for each movie
    average_ratings_weighted['WR'] = (
        (average_ratings_weighted['v'] / (average_ratings_weighted['v'] + m)) * average_ratings_weighted['U'] +
        (m / (average_ratings_weighted['v'] + m)) * C
    )

    # Return top N movieIds and scores
    ids = average_ratings_weighted['WR'].nlargest(N).index.tolist()
    scores = average_ratings_weighted['WR'].nlargest(N).values.tolist()
    return ids, scores


## Collaborative Filtering
def user_based_pearson_cf(df, target_user, N=10, k=30, min_common=3, shrink=10):
    # Build user–item matrix
    mat = df.pivot_table(
        index="userId", columns="movieId", values="rating", aggfunc="mean"
    ).astype(float)

    # Raise error if the target user isn’t in the matrix
    if target_user not in mat.index:
        raise ValueError(f"userId {target_user!r} not found.")

    # Compute user–user pair-wise Pearson similarities
    demeaned = mat.sub(mat.mean(axis=1), axis=0)
    sim = demeaned.T.corr(method="pearson", min_periods=min_common).fillna(0.0)

    # Predict ratings for unrated items
    unrated_items = mat.columns[mat.loc[target_user].isna()]
    r_u_bar = mat.loc[target_user].mean()
    preds = {}

    for item in unrated_items:
        # Users who have rated this item
        neighbours = mat.index[mat[item].notna()]
        if neighbours.empty:
            continue

        # Get similarities
        sims = sim.loc[target_user, neighbours]

        # Check top k most similar neighbours
        top_k = sims.abs().nlargest(k).index
        sims_k = sims.loc[top_k]
        ratings_k = mat.loc[top_k, item]
        means_k = mat.loc[top_k].mean(axis=1)

        # Compute the prediction
        numer = ((ratings_k - means_k) * sims_k).sum()
        denom = sims_k.abs().sum() + shrink

        # Save the prediction
        preds[item] = r_u_bar + numer / denom if denom else r_u_bar

    # Return top N movieIds and scores
    if not preds:
        return []
    ids = pd.Series(preds).nlargest(N).index.tolist()
    scores = pd.Series(preds).nlargest(N).values.tolist()
    return ids, scores

def user_based_cosine_cf(df, target_user, N=10, k=30, shrink=10):
    # Build user–item matrix
    mat = df.pivot_table(
        index="userId", columns="movieId", values="rating", aggfunc="mean"
    ).astype(float)

    # Raise error if the target user isn’t in the matrix
    if target_user not in mat.index:
        raise ValueError(f"userId {target_user!r} not in data.")

    # Compute user–user cosine similarity on filled mean-centered data
    X = mat.fillna(0.0).values
    sim = pd.DataFrame(
        cosine_similarity(X),
        index=mat.index,
        columns=mat.index
    ).astype(float)

    # Predict ratings for unrated items
    preds = {}
    unrated = mat.columns[mat.loc[target_user].isna()]

    for item in unrated:
        # Users who have rated this item
        neighbours = mat.index[mat[item].notna()]
        if neighbours.empty:
            continue

        # Get similarities
        scores = sim.loc[target_user, neighbours]
        topk = scores.abs().nlargest(k)
        if topk.empty:
            continue
        
        # Compute the prediction
        r_k = mat.loc[topk.index, item]
        w_k = topk
        denom = w_k.abs().sum() + shrink
        if denom>0:
            preds[item] = (w_k * r_k).sum() / denom

    # Return top N movieIds and scores
    if not preds:
        return []
    ids = pd.Series(preds).nlargest(N).index.tolist()
    scores = pd.Series(preds).nlargest(N).values.tolist()
    return ids, scores

def item_based_pearson_cf(df, target_user, N=10, k=30, min_common=3, shrink=10):
    # Build user–item matrix
    mat = df.pivot_table(
        index="userId", columns="movieId", values="rating", aggfunc="mean"
    ).astype(float)

    # Raise error if the target user isn’t in the matrix
    if target_user not in mat.index:
        raise ValueError(f"userId {target_user!r} not found.")

    # Compute user–user pair-wise Pearson similarities
    item_means = mat.mean(axis=0)
    demeaned = mat.sub(item_means, axis=1)
    sim = demeaned.corr(method="pearson", min_periods=min_common).fillna(0.0)

    # Predict ratings for unrated items
    preds = {}
    seen = mat.loc[target_user].dropna().index
    candidates = mat.columns[mat.loc[target_user].isna()]

    for item in candidates:
        # Check if the item exists in similarity matrix
        if item not in sim.index:
            continue

        # Users who have rated this item
        neighbours = seen.intersection(sim.index)
        if neighbours.empty:
            continue

        # Check top k most similar neighbours
        sims = sim.loc[item, neighbours]
        top_k = sims.abs().nlargest(k).index
        sims_k = sims.loc[top_k]

        # Compute the prediction
        r_k = mat.loc[target_user, top_k]
        m_k = item_means[top_k]
        numer = ((r_k - m_k) * sims_k).sum()
        denom = sims_k.abs().sum() + shrink

        # Save the prediction
        preds[item] = item_means[item] + (numer / denom if denom else 0.0)

    # Return top N movieIds and scores
    if not preds:
        return []
    ids = pd.Series(preds).nlargest(N).index.tolist()
    scores = pd.Series(preds).nlargest(N).values.tolist()
    return ids, scores

def item_based_cosine_cf(df, target_user, N=10, k=30, shrink=10):
    # Build user–item matrix
    mat = df.pivot_table(
        index="userId", columns="movieId", values="rating", aggfunc="mean"
    ).astype(float)

    # Raise error if the target user isn’t in the matrix
    if target_user not in mat.index:
        raise ValueError(f"userId {target_user!r} not in data.")

    # Compute user–user cosine similarity on filled mean-centered data
    X = mat.fillna(0.0).T.values
    sim = pd.DataFrame(
        cosine_similarity(X),
        index=mat.columns,
        columns=mat.columns
    ).astype(float)

    # Collect the user’s existing ratings
    user_ratings = mat.loc[target_user].dropna()
    if user_ratings.empty:
        return []

    # Predict ratings for unrated items
    preds = {}
    candidates = mat.columns[mat.loc[target_user].isna()]
    
    for item in candidates:
        # Get similarities
        scores = sim.loc[item, user_ratings.index]
        
        # Pick top k most similar neighbours
        topk = scores.abs().nlargest(k)
        if topk.empty:
            continue

        # Compute the prediction
        r_j = user_ratings[topk.index]
        w_j = topk
        denom = w_j.abs().sum() + shrink
        if denom > 0:
            preds[item] = (w_j * r_j).sum() / denom

    # Return top N movieIds and scores
    if not preds:
        return []
    ids = pd.Series(preds).nlargest(N).index.tolist()
    scores = pd.Series(preds).nlargest(N).values.tolist()
    return ids, scores


## Content-Based Filtering
def content_based_cosine_f(df, target_user, N=10):
    # Build movie-feature matrix from genres
    movies = (
        df[['movieId','genres']]
        .drop_duplicates()
        .assign(genres_list=lambda d: d['genres'].str.split('|'))
    )
    
    # Encode genres as binary features
    mlb = MultiLabelBinarizer()
    genre_mat = mlb.fit_transform(movies['genres_list'])

    # Apply TF-IDF transformation
    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
    tfidf_mat = tfidf.fit_transform(genre_mat)

    # Create DataFrame with movie features
    features = pd.DataFrame(
        tfidf_mat.toarray(),
        index=movies['movieId'],
        columns=mlb.classes_
    )

    # Build user profile as weighted average of their rated-movie features
    user_hist = (
        df[df['userId'] == target_user]
          .loc[:, ['movieId','rating']]
          .dropna(subset=['rating'])
    )
    if user_hist.empty:
        return []

    # Align features & ratings
    user_feats = features.loc[user_hist['movieId']].values
    ratings = user_hist['rating'].values.reshape(-1,1)

    # Weighted average as singular profile vector
    user_profile = (user_feats * ratings).sum(axis=0) / ratings.sum()

    # Score all unseen movies
    seen = set(user_hist['movieId'])
    candidates = [m for m in features.index if m not in seen]
    if not candidates:
        return []

    # Compute cosine similarity
    cand_feat = features.loc[candidates].values
    sims = cosine_similarity(cand_feat, user_profile.reshape(1,-1)).flatten()
    sims_series = pd.Series(sims, index=candidates)
    
    # Return top N movieIds and scores
    if sims_series.empty:
        return []
    ids = sims_series.nlargest(N).index.tolist()
    scores = sims_series.nlargest(N).values.tolist()
    return ids, scores


## Matrix Factorisation
def matrix_factorisation_svd(df, target_user, N=10, k=30, random_state=42):
    # Build user–item matrix
    R = (
        df.pivot_table(
            index="userId", columns="movieId", values="rating", aggfunc="mean"
        )
        .astype(float)
    )

    # Raise error if the target user isn’t in the matrix
    if target_user not in R.index:
        raise ValueError(f"userId {target_user!r} not found in data.")

    # Demean by user means
    user_means = R.mean(axis=1)
    R_demeaned = R.sub(user_means, axis=0).fillna(0.0)

    # Apply Model
    model = TruncatedSVD(n_components=k, random_state=random_state)
    user_factors = model.fit_transform(R_demeaned.values)
    item_factors = model.components_

    # Approximate ratings and add back means
    R_hat = np.dot(user_factors, item_factors)
    R_hat += user_means.values.reshape(-1, 1)

    # Save predictions
    preds_df = pd.DataFrame(R_hat, index=R.index, columns=R.columns)

    # Pick top N for the target user
    user_pred = preds_df.loc[target_user]
    seen = R.loc[target_user].dropna().index
    top_by_matrix_factorisation_svd = (
        user_pred.drop(seen).nlargest(N).index.tolist()
    )

    # Return top N movieIds and scores
    if not top_by_matrix_factorisation_svd:
        return []
    ids = top_by_matrix_factorisation_svd
    scores = user_pred[top_by_matrix_factorisation_svd].values.tolist()
    return ids, scores

def matrix_factorisation_nmf(df, target_user, N=10, k=30, random_state=42, max_iter=500):
    # Build user–item matrix
    R = (
        df.pivot_table(
            index="userId", columns="movieId", values="rating", aggfunc="mean"
        )
        .fillna(0.0)
    )

    # Raise error if the target user isn’t in the matrix
    if target_user not in R.index:
        raise ValueError(f"userId {target_user!r} not found in data.")

    # Apply Model
    model = NMF(n_components=k, init="random", random_state=random_state, max_iter=max_iter, tol=1e-4)
    user_factors = model.fit_transform(R.values)
    item_factors = model.components_

    # Reconstruct ratings matrix
    R_hat = np.dot(user_factors, item_factors)

    # Save predictions
    preds_df = pd.DataFrame(R_hat, index=R.index, columns=R.columns)

    # Pick top N for the target user
    user_pred = preds_df.loc[target_user]
    seen = R.loc[target_user].to_numpy().nonzero()[0]
    seen_ids = R.loc[target_user][R.loc[target_user] > 0].index
    top_by_matrix_factorisation_nmf = (
        user_pred.drop(seen_ids).nlargest(N).index.tolist()
    )

    # Return top N movieIds and scores
    if not top_by_matrix_factorisation_nmf:
        return []
    ids = top_by_matrix_factorisation_nmf
    scores = user_pred[top_by_matrix_factorisation_nmf].values.tolist()
    return ids, scores

def matrix_factorisation_als(df, target_user, N=10, factors=30, regularization=0.1, iterations=15, alpha=1.0):
    # Build id mappings
    unique_users = df['userId'].unique()
    unique_items = df['movieId'].unique()
    user2idx = {u: i for i, u in enumerate(unique_users)}
    item2idx = {m: i for i, m in enumerate(unique_items)}
    idx2item = {i: m for m, i in item2idx.items()}

    # Raise error if the target user isn’t in the matrix
    if target_user not in user2idx:
        raise ValueError(f"userId {target_user!r} not found in data.")

    # Build item-user confidence matrix
    rows = df['movieId'].map(item2idx).to_numpy()
    cols = df['userId'].map(user2idx).to_numpy()
    data = (1.0 + alpha * df['rating'].astype(float)).to_numpy()
    item_user_csr = coo_matrix(
        (data, (rows, cols)),
        shape=(len(unique_items), len(unique_users))
    ).tocsr()

    # Apply model
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        calculate_training_loss=False
    )
    model.fit(item_user_csr)

    # Get raw scores
    uidx = user2idx[target_user]
    user_vec = model.user_factors[uidx]
    item_vecs = model.item_factors
    scores_all = item_vecs.dot(user_vec)

    # Mask out already-rated items   
    seen_items = df.loc[df['userId'] == target_user, 'movieId'].unique()
    seen_idx = [item2idx[m] for m in seen_items if m in item2idx]
    for i in seen_idx:
        if 0 <= i < scores_all.shape[0]:
            scores_all[i] = -np.inf

    # Pick Top N
    top_idx = np.argpartition(-scores_all, N)[:N]
    top_idx = top_idx[np.argsort(-scores_all[top_idx])]

    # Return top N movieIds and scores
    ids = [idx2item[i] for i in top_idx]
    scores = scores_all[top_idx].tolist()
    return ids, scores


## Hybrid
# Z normalization
def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

# Min-Max normalization
def _minmax01(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / (rng + 1e-9)

def hybrid_recommender_cf_aw(df, target_user, N=10, alpha=0.8, min_interactions=3, k=30, shrink=10, m=10):
    # Collect user history
    interactions = df[df["userId"] == target_user]
    seen_items: set[int] = set(interactions["movieId"].unique())
    seen_count = len(seen_items)

    # Popularity prior – Bayesian weighted rating for every movie
    n_movies = df["movieId"].nunique()
    prior_ids, prior_scores = average_rating_weighted(df, N=n_movies, m=m)
    prior_series = pd.Series(prior_scores, index=prior_ids, name="prior")

    # Cold‑start path – rely entirely on the prior  
    if seen_count < min_interactions:
        print(f"Cold Start Detected: User {target_user} has only {seen_count} interactions, using non-personalized model.")
        top = prior_series[~prior_series.index.isin(seen_items)].nlargest(N)
        return pd.DataFrame(
            {
                "movieId": top.index,
                "score": _minmax01(top.values),
                "cf_raw": np.nan,
                "prior_raw": top.values,
                "alpha_used": 0.0,
                "seen_count": seen_count,
            }
        )

    # Personalised CF scores  
    cf_ids, cf_scores = user_based_cosine_cf(
        df,
        target_user,
        N=max(1000, n_movies),
        k=k,
        shrink=shrink,
    )
    cf_series = pd.Series(cf_scores, index=cf_ids, name="cf")

    # Blend the two models
    candidates = prior_series.index.union(cf_series.index).difference(seen_items)
    cf_aligned = cf_series.reindex(candidates)
    prior_aligned = prior_series.reindex(candidates)

    # Normalise to share scale before mixing
    cf_norm = _zscore(cf_aligned.fillna(cf_aligned.mean()))
    prior_norm = _zscore(prior_aligned)

    # Mix the two models
    final = alpha * cf_norm + (1 - alpha) * prior_norm
    final = _minmax01(final)

    # Return top N movieIds and scores
    ids = final.nlargest(N).index.tolist()
    scores = final.nlargest(N).values.tolist()
    return ids, scores

def hybrid_recommender(df, target_user, cold_model=average_rating_weighted, warm_model=user_based_cosine_cf, N=10, alpha=0.8, min_interactions=3):
    # Collect user history
    interactions = df[df["userId"] == target_user]
    seen_items: set[int] = set(interactions["movieId"].unique())
    seen_count = len(seen_items)

    # Apply cold model
    n_movies = df["movieId"].nunique()
    prior_ids, prior_scores = cold_model(df, N=n_movies)
    prior_series = pd.Series(prior_scores, index=prior_ids, name="prior")

    # Cold‑start path – rely entirely on the prior  
    if seen_count < min_interactions:
        print(f"Cold Start Detected: User {target_user} has only {seen_count} interactions, using non-personalized model.")
        top = prior_series[~prior_series.index.isin(seen_items)].nlargest(N)

        # Return top N movieIds and scores
        ids = top.index.tolist()
        scores = top.values.tolist()
        return ids, scores

    # Apply warm model
    cf_ids, cf_scores = warm_model(df, target_user, N=max(1000, n_movies))
    cf_series = pd.Series(cf_scores, index=cf_ids, name="cf")

    # Blend the two models
    candidates = prior_series.index.union(cf_series.index).difference(seen_items)
    cf_aligned = cf_series.reindex(candidates)
    prior_aligned = prior_series.reindex(candidates)

    # Normalise to share scale before mixing
    cf_norm = _zscore(cf_aligned.fillna(cf_aligned.mean()))
    prior_norm = _zscore(prior_aligned)

    # Mix the two models
    final = alpha * cf_norm + (1 - alpha) * prior_norm
    final = _minmax01(final)

    # Return top N movieIds and scores
    ids = final.nlargest(N).index.tolist()
    scores = final.nlargest(N).values.tolist()
    return ids, scores


## Evaluation
def make_train_test(df, test_size=0.2):
    # Hold out the most recent X% of interactions
    df_sorted = df.sort_values("timestamp")
    cutoff = int(len(df) * (1 - test_size))
    train = df_sorted.iloc[:cutoff]
    test  = df_sorted.iloc[cutoff:]
    return train.reset_index(drop=True), test.reset_index(drop=True)

def evaluate_top_n(df_train, df_test, model_func, N=10):
    # Build ground‐truth list
    gt = df_test.groupby('userId')['movieId'].apply(list).to_dict()
    
    # Calculate metrics
    precisions, recalls, mrrs = [], [], []
    train_users = set(df_train['userId'].unique())
    
    # Process each user in the test set
    for user, actual in gt.items():
        if not actual or user not in train_users:
            continue
            
        try:
            # Get recommendations
            recs_and_scores = model_func(df_train, target_user=user, N=N)
            
            # Handle both return types (older functions might just return IDs)
            if isinstance(recs_and_scores, tuple) and len(recs_and_scores) == 2:
                recs, scores = recs_and_scores
            else:
                continue
                
            if not recs:
                continue
                
            # Calculate ranking metrics
            precisions.append(len(set(recs[:N]) & set(actual)) / N)
            recalls.append(len(set(recs[:N]) & set(actual)) / len(actual))
            
            # MRR calculation
            rank = next((i+1 for i, r in enumerate(recs[:N]) if r in actual), 0)
            mrrs.append(1/rank if rank > 0 else 0)
            
        except Exception:
            # Skip problematic users
            continue
    
    # Return metrics dictionary
    return {
        f'Precision@{N}': np.nanmean(precisions) if precisions else np.nan,
        f'Recall@{N}': np.nanmean(recalls) if recalls else np.nan,
        f'MRR@{N}': np.nanmean(mrrs) if mrrs else np.nan,
    }

def evaluate_rmse(df_train, df_test, model_func, N=10):
    # Create a dictionary mapping (userId, movieId) to rating
    test_ratings = df_test.set_index(['userId', 'movieId'])['rating'].to_dict()
    
    # Initialize a list to collect errors
    all_errors = []
    train_users = set(df_train['userId'].unique())
    
    # Process each user in the test set
    for user in df_test['userId'].unique():
        if user not in train_users:
            continue
            
        try:
            # Get recommendations and scores
            recs_and_scores = model_func(df_train, target_user=user, N=N)
            
            # Handle different return types
            if isinstance(recs_and_scores, tuple) and len(recs_and_scores) == 2:
                recs, scores = recs_and_scores
            else:
                continue
                
            if not recs:
                continue
                
            # Find items that exist in both recommendations and test set
            user_errors = []
            for i, item in enumerate(recs):
                if (user, item) in test_ratings:
                    actual = test_ratings[(user, item)]
                    predicted = scores[i]  
                    user_errors.append((actual - predicted) ** 2)
            
            if user_errors:  # Only add if we have overlapping items
                all_errors.extend(user_errors)
                
        except Exception:
            # Skip problematic users
            continue
    
    # Return RMSE
    return np.sqrt(np.mean(all_errors)) if all_errors else np.nan

def evaluate_models(df_train, df_test, models, N=10):
    # Initialize results dictionary
    results = {}
    
    # Iterate over each model
    for name, fn in models.items():
        
        try:
            # Evaluate top-N metrics
            topn_metrics = evaluate_top_n(df_train, df_test, fn, N=N)
            
            # Evaluate RMSE (if applicable)
            rmse = evaluate_rmse(df_train, df_test, fn, N=N)
            
            # Store results
            results[name] = {
                'TopN': topn_metrics,
                'RMSE': rmse
            }
            
            # Display results
            metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in topn_metrics.items()])
            print(f"{name} - {metrics_str}, RMSE: {rmse:.4f}")
            
        except Exception as e:
            results[name] = {'error': str(e)}
    
    return results
