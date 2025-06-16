### The Streamlit application for my Recommender System
import streamlit as st
import pandas as pd
import numpy as np
import io
import contextlib
import rec_sys_algos as rs


# ---- 1. ALGORITHMS ----

# Define the algorithms and their descriptions
ALGOS = {
    "Top-N by count": (rs.top_n_count, False, "Returns the top-N most rated movies based on the frequency of ratings in the dataset."),
    "Top-N by likes": (rs.top_n_likes, False, "Returns the top-N movies based on the number of likes (ratings >= 4)."),
    "Top-N by average rating": (rs.average_rating, False, "Returns the top-N movies based on the average rating."),
    "Top-N by average rating (normalized)": (rs.average_rating_normalized, False, "Ranks the top-N movies by their normalized average rating, adjusting for user rating biases."),
    "Top-N by average rating (weighted)": (rs.average_rating_weighted, False, "Recommends the top-N movies using a weighted average that balances movie rating and rating count."),
    "User-based CF (Pearson)": (rs.user_based_pearson_cf, True, "Generates top-N personalized movie recommendations using user-based collaborative filtering with Pearson similarity."),
    "User-based CF (Cosine)": (rs.user_based_cosine_cf, True, "Recommends top-N movies using user-based collaborative filtering with cosine similarity and shrinkage."),
    "Item-based CF (Pearson)": (rs.item_based_pearson_cf, True, "Generates top-N movie recommendations using item-based collaborative filtering with Pearson similarity."),
    "Item-based CF (Cosine)": (rs.item_based_cosine_cf, True, "Recommends top-N movies using item-based collaborative filtering with cosine similarity and shrinkage."),
    "Content-based (Cosine)": (rs.content_based_cosine_f, True, "Recommends top-N movies by matching user preferences to genre-based TF-IDF profiles using cosine similarity."),
    "Matrix-factorisation (SVD)": (rs.matrix_factorisation_svd, True, "Recommends top-N movies using matrix factorization via SVD to predict ratings from latent user-item factors."),
    "Matrix-factorisation (NMF)": (rs.matrix_factorisation_nmf, True, "Recommends top-N movies using Non-negative Matrix Factorization to uncover latent user-item preferences."),
    "Matrix-factorisation (ALS)": (rs.matrix_factorisation_als, True, "Recommends top-N movies using Alternating Least Squares (ALS) matrix factorization on implicit feedback data."),
    "Recommended Hybrid (weighted average & user-based cosine)": (rs.hybrid_recommender_cf_aw, True, "Recommends top-N movies by blending personalized collaborative filtering with a popularity-based weighted average using a weighted hybrid approach."),
    "Custom Hybrid": (rs.hybrid_recommender, True, "Recommends top-N movies by adaptively blending a cold-start and warm-start model from user input.")
}

# Function to run the selected algorithm
def run_algo(name, df, user, top_n, **kwargs):
    fn, needs_user, _ = ALGOS[name]
    if name == "Custom Hybrid":
        ids, scores = fn(
            df,
            target_user          = user,
            N                    = top_n,
            cold_model           = kwargs.get("cold_model"),
            warm_model           = kwargs.get("warm_model"),
            alpha                = kwargs.get("alpha", 0.8),
            min_interactions     = kwargs.get("min_interactions", 3),
        )
    elif needs_user:
        ids, scores = fn(df, target_user=user, N=top_n)
    else:
        ids, scores = fn(df, N=top_n)
    return pd.DataFrame({"movieId": ids, "score": scores})


# ---- 2. DATA ----

# Load and cache the data
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("data/merged.csv")
    ratings = df[["userId", "movieId", "rating", "timestamp"]]
    movies = df[["movieId", "title", "genres"]].drop_duplicates("movieId").set_index("movieId")
    return ratings, movies
ratings, movies = load_data()


# ---- 3. SIDEBAR ----

# Set the sidebar title and description
st.sidebar.header("⚙️ Settings")

# Select the algorithm and its parameters
st.sidebar.subheader("🔀 Algorithm")
algo_name = st.sidebar.selectbox("Available Models", list(ALGOS))
top_n = st.sidebar.slider("No. of recommendations", 5, 40, 10)

# Display the description of the selected algorithm
st.sidebar.markdown(f"**{algo_name}**: {ALGOS[algo_name][2]}")

# Check if the custom hybrid recommender is selected
hybrid_params = {}
if algo_name == "Custom Hybrid":
    st.sidebar.markdown("🔧 Hybrid settings")

    # Split ALGOS into cold (no user) and warm (needs user) pools
    cold_models = {n: t[0]  for n, t in ALGOS.items() if not t[1]}

    warm_models = {n: t[0]  for n, t in ALGOS.items()
                   if t[1] and n not in {"Custom Hybrid", "Recommended Hybrid (weighted average & user-based cosine)"}}


    # Select alpha blend weight
    alpha = st.sidebar.slider("Blend weight α (% warm-model used)", 0.0, 1.0, 0.8, 0.05)

    # Select switch threshold
    min_int = st.sidebar.number_input(
        "Min. interactions before CF kicks in",
        min_value=1, max_value=50, value=3, step=1
    )

    # Select models to combine
    cold_choice = st.sidebar.selectbox(
        "Cold-start model (non-personalised)", list(cold_models))
    warm_choice = st.sidebar.selectbox(
        "Warm model (needs user)", list(warm_models))

    # Save parameters
    hybrid_params.update({
        "cold_model"      : cold_models[cold_choice],
        "warm_model"      : warm_models[warm_choice],
        "alpha"           : alpha,
        "min_interactions": min_int,
    })

# Check if the selected algorithm needs a user ID
needs_user = ALGOS[algo_name][1]
if needs_user:
    st.sidebar.subheader("🎯 User ID")

    # Choose input mode
    use_random = st.sidebar.checkbox("Pick a random user from the dataset", value=False)
    valid_ids = ratings.userId.unique()
    
    # Random input­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­ mode
    if use_random:
        user_id = int(ratings.userId.sample(1).iloc[0])

    # Manual input mode
    else:
        user_id = int(
            st.sidebar.number_input(
                "Target user ID",
                min_value=int(valid_ids.min()),
                max_value=int(valid_ids.max()),
                value=int(valid_ids.min()),
                step=1,
            )
        )
        if user_id not in valid_ids:
            st.sidebar.warning(
                f"User {user_id} not found. Defaulting to {valid_ids[0]}"
            )
            user_id = int(valid_ids[0])

else:
    user_id = None

# Add a button to generate recommendations
run = st.sidebar.button("🚀 Generate", use_container_width=True)


# ---- 4. MAIN PANE  ----

# Set the main title and description
st.title("🎬 MovieLens Recommender")
st.markdown(
    """Select an algorithm on the left, tweak the parameters and hit **Generate**.
    The table below will refresh with the top-N items and raw scores."""
)

# Split the main pane into two tabs
rec_tab, eval_tab = st.tabs(["🎯 Recommendations", "📈 Live evaluation"])

with rec_tab:
    if run:
        # Add a loading spinner
        with st.spinner("Generating recommendations..."):
            
            try:
                # Create a buffer to capture the log output
                log_buffer = io.StringIO()
                with contextlib.redirect_stdout(log_buffer):
                    
                    # Run the selected algorithm
                    recs = run_algo(algo_name, ratings, user_id, top_n, **hybrid_params)
                
                # Display the log output if any
                log_text = log_buffer.getvalue().strip()
                if log_text:
                    st.info(log_text)

                # Save the generated recommendations
                recs = (
                    recs.merge(
                        movies[["title", "genres"]],
                        left_on = "movieId",
                        right_index = True,
                        how = "left"
                    )
                )
                
                # Display the recommendations
                st.success(f"{len(recs)} recommendations for user **{user_id}** via **{algo_name}**")
                st.dataframe(
                    recs[["movieId", "title", "genres", "score"]].round(3),
                    use_container_width=True,
                    hide_index=True
                )

            # Raise any exceptions that occur during the recommendation generation
            except Exception as e:
                st.error(f"⚠️ {e}")


# ---- 5. LIVE EVALUATION TAB ----

with eval_tab:
    st.markdown(
        "Choose a test hold-out size, pick one or more models and click **Evaluate** "
        "to compute Precision / Recall / MRR and RMSE on the last portion of the "
        "ratings (chronological split)."
    )

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test size (% used for test)", 0.05, 0.5, 0.2, 0.05)

    with col2:
        top_n_eval = st.slider("N for Precision/Recall/MRR", 5, 50, 10, 5)

    models_to_eval = st.multiselect(
        "Models to evaluate",
        options=list(ALGOS.keys()),
        default=[algo_name]
    )

    # Add a button to generate evaluations
    evaluate = st.button("🏁 Evaluate", use_container_width=True)

    if evaluate:
        st.divider()

        train, test = rs.make_train_test(ratings, test_size=test_size)

        num_models = len(models_to_eval)
        progress_bar = st.progress(0, text="Starting evaluation...")

        rows = []
        metric_cols = set()

        for idx, model_name in enumerate(models_to_eval, 1):
            fn = ALGOS[model_name][0]

            topn = rs.evaluate_top_n(train, test, fn, N=top_n_eval)
            rmse = rs.evaluate_rmse(train, test, fn, N=top_n_eval)

            metric_cols.update(topn.keys())
            rows.append({**{"Model": model_name}, **topn, **{"RMSE": rmse}})

            progress_bar.progress(idx / num_models,
                                text=f"Finished {idx}/{num_models} models")

        metric_cols = sorted(metric_cols)
        res_df = (pd.DataFrame(rows)
                    .set_index("Model")
                    [metric_cols + ["RMSE"]]
                    .round(4))

        st.subheader("📊 Metrics")
        st.dataframe(res_df, use_container_width=True)