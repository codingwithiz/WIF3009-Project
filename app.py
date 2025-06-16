import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import os
import json
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import math

# Set page configuration
st.set_page_config(
    page_title="Influencer Social Network Analysis Dashboard", layout="wide"
)

# Title and description
st.title("🌐 Influencer Social Network Analysis Dashboard")
st.markdown(
    """
This dashboard visualizes social network analysis, engagement prediction, topic modeling, and sentiment analysis for top influencers, including:
- Network graph with nodes sized by influencer score and colored by community.
- Bar plot comparing centrality metrics across influencers.
- Scatter plot of followers count vs. influencer score.
- Feature importance for engagement prediction.
- Actual vs. predicted engagement for influencers.
- Average engagement rate by day of week.
- Optimal tweeting times by influencer (table and heatmaps).
- Engagement by tweet topic with top keywords.
- Tweet sentiment distribution and correlation with engagement.
- Tables of top influencers, community statistics, and optimal tweeting times.
- Challenges and future work for the analysis.
"""
)


# Function to preprocess raw tweet data
def preprocess(df):
    if df.empty:
        st.warning("No data available for preprocessing.")
        return df

    df = df.dropna(subset=["text", "created_at"])
    df["text_length"] = df["text"].str.len()
    df["week"] = df["created_at"].dt.to_period("W").astype(str)
    df["day_of_week"] = df["created_at"].dt.dayofweek
    df["hour"] = df["created_at"].dt.hour
    df["total_engagement"] = (
        df["retweet_count"] + df["reply_count"] + df["like_count"] + df["quote_count"]
    )
    return df


# Function to load raw tweet data from individual JSON files
@st.cache_data
def load_tweet_data(tweet_folder="data/raw/tweets/"):
    try:
        if not os.path.exists(tweet_folder):
            st.error(f"Directory {tweet_folder} does not exist.")
            return pd.DataFrame()

        records = []
        for filename in os.listdir(tweet_folder):
            if not filename.endswith(".json"):
                continue
            username = filename.replace(".json", "")
            with open(os.path.join(tweet_folder, filename), "r") as f:
                data = json.load(f)
                instructions = (
                    data[0]
                    .get("tweets", {})
                    .get("result", {})
                    .get("timeline", {})
                    .get("instructions", [])
                )
                for instruction in instructions:
                    if "entries" not in instruction:
                        continue
                    for entry in instruction["entries"]:
                        tweet = (
                            entry.get("content", {})
                            .get("itemContent", {})
                            .get("tweet_results", {})
                            .get("result", {})
                        )
                        legacy = tweet.get("legacy", {})
                        if not legacy:
                            continue
                        mentions = []
                        entities = legacy.get("entities", {})
                        if "user_mentions" in entities:
                            mentions = [
                                m.get("screen_name", "")
                                for m in entities["user_mentions"]
                            ]
                        in_reply_to = legacy.get("in_reply_to_screen_name", None)
                        records.append(
                            {
                                "username": username,
                                "created_at": legacy.get("created_at"),
                                "retweet_count": legacy.get("retweet_count", 0),
                                "reply_count": legacy.get("reply_count", 0),
                                "like_count": legacy.get("favorite_count", 0),
                                "quote_count": legacy.get("quote_count", 0),
                                "text": legacy.get("full_text", ""),
                                "mentions": mentions,
                                "in_reply_to": in_reply_to,
                                "text_length": len(legacy.get("full_text", "")),
                            }
                        )
        df = pd.DataFrame(records)
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        if df.empty:
            st.warning("No tweet data loaded from data/raw/tweets/")
            return df
        # Apply preprocessing to ensure required columns
        df = preprocess(df)
        return df
    except Exception as e:
        st.error(f"Error loading tweet data: {e}")
        return pd.DataFrame()


# Function to load tweets_data.json for topic modeling and sentiment
@st.cache_data
def load_tweets_data_json(json_file="data/raw/tweets_data.json"):
    try:
        with open(json_file, "r") as f:
            tweets = json.load(f)

        rows = []
        for user in tweets:
            instructions = (
                user.get("tweets", {})
                .get("result", {})
                .get("timeline", {})
                .get("instructions", [])
            )
            for instruction in instructions:
                if instruction.get("type") == "TimelineAddEntries":
                    entries = instruction.get("entries", [])
                    for entry in entries:
                        content = entry.get("content", {})
                        item_content = content.get("itemContent", {})
                        tweet_result = item_content.get("tweet_results", {}).get(
                            "result", {}
                        )
                        legacy = tweet_result.get("legacy", {})
                        text = legacy.get("full_text", "")
                        timestamp = legacy.get("created_at", None)
                        if text and timestamp:
                            try:
                                dt = datetime.strptime(
                                    timestamp, "%a %b %d %H:%M:%S %z %Y"
                                )
                                rows.append(
                                    {
                                        "username": user.get("username", "unknown"),
                                        "text": text,
                                        "created_at": dt,
                                        "hour": dt.hour,
                                        "likes": legacy.get("favorite_count", 0),
                                        "retweets": legacy.get("retweet_count", 0),
                                        "reply_count": legacy.get("reply_count", 0),
                                        "quote_count": legacy.get("quote_count", 0),
                                    }
                                )
                            except Exception as e:
                                st.warning(f"Skipping tweet due to error: {e}")
                                continue

        df = pd.DataFrame(rows)
        df["total_engagement"] = (
            df["likes"] + df["retweets"] + df["reply_count"] + df["quote_count"]
        )
        if df.empty:
            st.warning("No tweets loaded from tweets_data.json")
        st.write(f"Loaded {len(df)} tweets from {json_file}")
        return df
    except FileNotFoundError:
        st.error(f"{json_file} not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading tweets_data.json: {e}")
        return pd.DataFrame()


# Function to aggregate weekly
def aggregate_weekly(df):
    grouped = (
        df.groupby(["username", "week"])
        .agg(
            {
                "total_engagement": "mean",
                "text_length": "mean",
                "retweet_count": "mean",
                "reply_count": "mean",
                "like_count": "mean",
                "quote_count": "mean",
                "text": "count",
            }
        )
        .rename(columns={"text": "tweet_count"})
        .reset_index()
    )
    return grouped


# Function to add rolling features
def add_rolling_features(df_weekly, window=3):
    df_weekly = df_weekly.sort_values(["username", "week"]).copy()
    df_weekly["rolling_engagement"] = df_weekly.groupby("username")[
        "total_engagement"
    ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    df_weekly["rolling_tweet_count"] = df_weekly.groupby("username")[
        "tweet_count"
    ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    df_weekly["engagement_per_text_length"] = df_weekly["total_engagement"] / df_weekly[
        "text_length"
    ].replace(0, 1)
    df_weekly["tweet_count_delta"] = (
        df_weekly.groupby("username")["tweet_count"].diff().fillna(0)
    )
    return df_weekly


# Function to generate tweets.csv
def generate_tweets_csv():
    st.info("Generating data/processed/tweets.csv...")
    tweet_folder = "data/raw/tweets/"
    raw_df = load_tweet_data(tweet_folder)
    if raw_df.empty:
        st.error(
            "Failed to load tweet data. Ensure JSON files exist in data/raw/tweets/."
        )
        return None

    weekly_df = aggregate_weekly(raw_df)
    st.write(f"Aggregated to {len(weekly_df)} weekly records.")

    weekly_df = add_rolling_features(weekly_df)

    try:
        centrality_df = pd.read_csv("data/outputs/centrality/influencer_centrality.csv")
    except FileNotFoundError:
        st.error("data/outputs/centrality/influencer_centrality.csv not found.")
        return None

    weekly_df = weekly_df.merge(
        centrality_df[
            [
                "screen_name",
                "degree_centrality",
                "pagerank",
                "betweenness_centrality",
                "closeness_centrality",
                "eigenvector_centrality",
                "followers_count",
                "friends_count",
                "statuses_count",
                "community",
            ]
        ],
        how="left",
        left_on="username",
        right_on="screen_name",
    )

    unmatched = weekly_df[weekly_df["screen_name"].isna()]["username"].unique()
    if len(unmatched) > 0:
        st.warning(
            f"{len(unmatched)} usernames not matched with centrality_df: {unmatched}"
        )

    weekly_df["community"] = weekly_df["community"].astype("category").cat.codes
    weekly_df["log_followers"] = np.log1p(weekly_df["followers_count"])
    weekly_df["engagement_per_follower"] = weekly_df["total_engagement"] / weekly_df[
        "followers_count"
    ].replace(0, 1)
    weekly_df.drop(columns=["screen_name"], inplace=True)

    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/tweets.csv"
    weekly_df.to_csv(output_path, index=False)
    st.success(f"Saved {len(weekly_df)} rows to {output_path}")
    return weekly_df


# Load all data
@st.cache_data
def load_data():
    try:
        centrality_df = pd.read_csv("data/outputs/centrality/influencer_centrality.csv")
        required_centrality_cols = [
            "user_id",
            "screen_name",
            "degree_centrality",
            "betweenness_centrality",
            "closeness_centrality",
            "eigenvector_centrality",
            "pagerank",
            "followers_count",
            "friends_count",
            "statuses_count",
            "community",
        ]
        if not all(col in centrality_df.columns for col in required_centrality_cols):
            st.error(
                f"Missing required columns in centrality_df: {set(required_centrality_cols) - set(centrality_df.columns)}"
            )
            return None, None, None, None, None

        G = nx.read_gexf("data/processed/influencer_network.gexf")

        tweets_csv = "data/processed/tweets.csv"
        if not os.path.exists(tweets_csv):
            st.warning("tweets.csv not found. Generating now...")
            weekly_df = generate_tweets_csv()
        else:
            weekly_df = pd.read_csv(tweets_csv)

        required_weekly_cols = [
            "username",
            "week",
            "total_engagement",
            "text_length",
            "tweet_count",
            "rolling_engagement",
            "rolling_tweet_count",
            "engagement_per_text_length",
            "tweet_count_delta",
            "degree_centrality",
            "pagerank",
            "betweenness_centrality",
            "closeness_centrality",
            "eigenvector_centrality",
            "followers_count",
            "friends_count",
            "statuses_count",
            "community",
            "log_followers",
            "engagement_per_follower",
        ]
        if not all(col in weekly_df.columns for col in required_weekly_cols):
            st.error(
                f"Missing required columns in weekly_df: {set(required_weekly_cols) - set(weekly_df.columns)}"
            )
            return None, None, None, None, None

        raw_df = load_tweet_data()
        required_raw_cols = [
            "username",
            "created_at",
            "total_engagement",
            "day_of_week",
            "hour",
        ]
        if not all(col in raw_df.columns for col in required_raw_cols):
            st.error(
                f"Missing required columns in raw_df: {set(required_raw_cols) - set(raw_df.columns)}"
            )
            return None, None, None, None, None

        df_tweets = load_tweets_data_json()
        required_tweets_cols = [
            "username",
            "text",
            "created_at",
            "hour",
            "likes",
            "retweets",
            "total_engagement",
        ]
        if not all(col in df_tweets.columns for col in required_tweets_cols):
            st.error(
                f"Missing required columns in df_tweets: {set(required_tweets_cols) - set(df_tweets.columns)}"
            )
            return None, None, None, None, None

        st.write(
            f"Loaded {len(centrality_df)} rows in centrality_df, {G.number_of_nodes()} nodes in graph, "
            f"{len(weekly_df)} rows in weekly_df, {len(raw_df)} rows in raw_df, {len(df_tweets)} rows in df_tweets"
        )
        return centrality_df, G, weekly_df, raw_df, df_tweets
    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None


# Function to plot influencer network
def plot_influencer_network(G, centrality_df):
    try:
        plt.figure(figsize=(12, 10))
        # Use spring layout with adjusted k to spread nodes and reduce overlap
        pos = nx.spring_layout(G, seed=42, k=0.5)
        influencer_scores = centrality_df.set_index("user_id")[
            "influencer_score"
        ].to_dict()
        min_score = min(influencer_scores.values()) if influencer_scores else 0
        node_sizes = [
            (influencer_scores.get(node, 0) + abs(min_score) + 1) * 2000
            for node in G.nodes()
        ]
        node_colors = []

        # Assign community colors
        for node in G.nodes():
            matching_rows = centrality_df.loc[
                centrality_df["user_id"] == node, "community"
            ]
            if not matching_rows.empty:
                node_colors.append(matching_rows.values[0])
            else:
                node_colors.append(0)

        # Draw nodes and edges
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.7,
            cmap=plt.cm.Set2,
        )
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

        # Create labels dictionary, prioritizing G's screen_name attribute
        labels = {}
        missing_labels = []
        for node in G.nodes():
            # Try getting screen_name from G's node attributes
            screen_name = G.nodes[node].get("screen_name", None)
            if screen_name is None:
                # Fallback to centrality_df
                matching_rows = centrality_df.loc[
                    centrality_df["user_id"] == node, "screen_name"
                ]
                if not matching_rows.empty:
                    screen_name = matching_rows.values[0]
                else:
                    screen_name = str(node)  # Fallback to user_id as string
                    missing_labels.append(node)
            labels[node] = screen_name

        if missing_labels:
            st.warning(
                f"Missing screen_name for {len(missing_labels)} nodes: {missing_labels}"
            )

        # Draw labels with improved visibility
        nx.draw_networkx_labels(
            G,
            pos,
            labels,
            font_size=12,
            font_color="white",
            font_weight="bold",
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
        )

        plt.title("Influencer Network - Centrality & Community", fontsize=16)
        plt.axis("off")
        plt.tight_layout()
        return plt.gcf()
    except Exception as e:
        st.error(f"Error in influencer network plot: {e}")
        return None


# Function to plot centrality comparison
def plot_centrality_comparison(centrality_df):
    try:
        plt.figure(figsize=(14, 8))
        measure_cols = [
            "degree_centrality",
            "betweenness_centrality",
            "closeness_centrality",
            "eigenvector_centrality",
            "pagerank",
        ]
        colors = sns.color_palette("husl", len(measure_cols))
        sorted_df = centrality_df.sort_values("influencer_score", ascending=False)
        x = np.arange(len(sorted_df))
        width = 0.15

        for i, column in enumerate(measure_cols):
            plt.bar(
                x + i * width, sorted_df[column], width, label=column, color=colors[i]
            )

        plt.xlabel("Influencers", fontsize=12)
        plt.ylabel("Centrality Value", fontsize=12)
        plt.title("Centrality Comparison by Influencer", fontsize=16)
        plt.xticks(x + width * 2, sorted_df["screen_name"], rotation=45, ha="right")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        return plt.gcf()
    except Exception as e:
        st.error(f"Error in centrality comparison plot: {e}")
        return None


# Function to plot followers vs influencer score
def plot_followers_vs_score(centrality_df):
    try:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x="followers_count",
            y="influencer_score",
            hue="community",
            size="pagerank",
            sizes=(100, 1000),
            data=centrality_df,
            palette="Spectral",
        )

        for i, row in centrality_df.iterrows():
            plt.annotate(
                row["screen_name"],
                (row["followers_count"], row["influencer_score"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        plt.title("Followers vs Influencer Score", fontsize=16)
        plt.xlabel("Followers Count", fontsize=12)
        plt.ylabel("Influencer Score", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        return plt.gcf()
    except Exception as e:
        st.error(f"Error in followers vs score plot: {e}")
        return None


# Function to plot feature importance
def plot_feature_importance(weekly_df):
    feature_cols = [
        "text_length",
        "degree_centrality",
        "betweenness_centrality",
        "closeness_centrality",
        "eigenvector_centrality",
        "pagerank",
        "log_followers",
        "friends_count",
        "statuses_count",
        "community",
        "rolling_engagement",
        "rolling_tweet_count",
        "engagement_per_text_length",
        "engagement_per_follower",
        "tweet_count_delta",
    ]
    try:
        df_model = weekly_df.dropna(subset=feature_cols + ["total_engagement"])
        if df_model.empty:
            st.warning(
                "No data available for feature importance after dropping NA values."
            )
            return None
        X = df_model[feature_cols]
        y = df_model["total_engagement"]

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(
            ascending=False
        )

        plt.figure(figsize=(8, 4))
        sns.barplot(x=feat_imp.values, y=feat_imp.index)
        plt.title("Feature Importance for Engagement Prediction")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        return plt.gcf()
    except Exception as e:
        st.error(f"Error in feature importance plot: {e}")
        return None


# Function to plot combined influencer predictions
def plot_combined_influencer_predictions(weekly_df, test_weeks=4):
    try:
        influencers = [
            inf
            for inf in weekly_df["username"].unique()
            if len(weekly_df[weekly_df["username"] == inf]) >= test_weeks + 3
        ]
        if not influencers:
            st.warning("No influencers with enough data to plot predictions.")
            return None

        n = len(influencers)
        cols = 2
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), squeeze=False)

        for i, influencer in enumerate(influencers):
            df = (
                weekly_df[weekly_df["username"] == influencer]
                .sort_values("week")
                .copy()
            )
            df["week_num"] = range(len(df))
            X = df[["week_num"]]
            y = df["total_engagement"]
            X_train, y_train = X[:-test_weeks], y[:-test_weeks]
            X_test, y_test = X[-test_weeks:], y[-test_weeks:]

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            ax = axes[i // cols][i % cols]
            ax.plot(df["week"], y, label="Actual", marker="o")
            ax.plot(
                df["week"].iloc[-test_weeks:], y_pred, label="Predicted", marker="x"
            )
            ax.set_title(f"{influencer}")
            ax.set_xlabel("Week")
            ax.set_ylabel("Engagement")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True)
            ax.legend()
            ax.text(
                0.95,
                0.95,
                f"MAE: {mean_absolute_error(y_test, y_pred):.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", fc="white", ec="gray"),
            )

        for j in range(i + 1, rows * cols):
            fig.delaxes(axes[j // cols][j % cols])

        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error in combined influencer predictions plot: {e}")
        return None


# Function to plot average engagement by day of week
def plot_engagement_by_day(raw_df):
    try:
        engagement_by_day = (
            raw_df.groupby("day_of_week")["total_engagement"].mean().reset_index()
        )
        engagement_by_day["day_name"] = engagement_by_day["day_of_week"].map(
            {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        )

        plt.figure(figsize=(8, 4))
        sns.barplot(x="day_name", y="total_engagement", data=engagement_by_day)
        plt.title("Average Engagement Rate by Day of Week")
        plt.ylabel("Avg Engagement")
        plt.xlabel("Day of Week")
        plt.tight_layout()
        return plt.gcf()
    except Exception as e:
        st.error(f"Error in engagement by day plot: {e}")
        return None


# Function to get optimal tweeting times
def get_optimal_tweeting_times(raw_df):
    try:
        best_times = (
            raw_df.groupby(["username", "day_of_week", "hour"])["total_engagement"]
            .mean()
            .reset_index()
        )
        optimal_times = best_times.loc[
            best_times.groupby("username")["total_engagement"].idxmax()
        ]
        optimal_times["day_name"] = optimal_times["day_of_week"].map(
            {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        )
        return (
            optimal_times[["username", "day_name", "hour", "total_engagement"]],
            best_times,
        )
    except Exception as e:
        st.error(f"Error in optimal tweeting times: {e}")
        return pd.DataFrame(), pd.DataFrame()


# Function to plot optimal tweeting time heatmaps (FacetGrid)
def plot_optimal_tweeting_heatmap_facet(best_times):
    try:
        day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        best_times["day_name"] = best_times["day_of_week"].map(day_map)
        top_influencers = best_times["username"].value_counts().nlargest(20).index
        filtered = best_times[best_times["username"].isin(top_influencers)]

        g = sns.FacetGrid(filtered, col="username", col_wrap=3, height=4, aspect=1.5)
        g.map_dataframe(
            lambda data, color: sns.heatmap(
                data.pivot(index="day_name", columns="hour", values="total_engagement"),
                cmap="YlGnBu",
                cbar=True,
                linewidths=0.5,
            )
        )
        g.set_titles(col_template="{col_name}")
        g.set_axis_labels("Hour", "Day of Week")
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle("Optimal Tweeting Time by Influencer (Avg Engagement)")
        return g.fig
    except Exception as e:
        st.error(f"Error in optimal tweeting heatmap (FacetGrid): {e}")
        return None


# Function to plot individual optimal tweeting heatmaps
def plot_optimal_tweeting_heatmaps_individual(best_times):
    try:
        influencers = best_times["username"].unique()
        n = len(influencers)
        cols = 3
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 6, rows * 5), sharex=True, sharey=True
        )
        axes = axes.flatten()

        for i, influencer in enumerate(influencers):
            data = best_times[best_times["username"] == influencer]
            pivot = data.pivot(
                index="hour", columns="day_of_week", values="total_engagement"
            )

            sns.heatmap(
                pivot, cmap="YlGnBu", annot=True, fmt=".1f", ax=axes[i], cbar=False
            )
            axes[i].set_title(f"{influencer}", pad=15)
            axes[i].set_ylabel("Hour of Day", labelpad=10)
            axes[i].set_xlabel("Day of Week (0=Mon)", labelpad=10)

        for j in range(i + 1, rows * cols):
            fig.delaxes(axes[j])

        fig.subplots_adjust(
            left=0.05, right=0.95, top=0.92, bottom=0.08, hspace=0.4, wspace=0.3
        )
        return fig
    except Exception as e:
        st.error(f"Error in individual optimal tweeting heatmaps: {e}")
        return None


# Function to perform topic modeling and plot engagement by topic
def plot_engagement_by_topic(df_tweets):
    try:
        vectorizer = CountVectorizer(stop_words="english", max_features=1000)
        X_topics = vectorizer.fit_transform(df_tweets["text"])

        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(X_topics)

        df_tweets["topic"] = lda.transform(X_topics).argmax(axis=1)
        topic_engagement = (
            df_tweets.groupby("topic")[["likes", "retweets"]]
            .mean()
            .sort_values(by="likes", ascending=False)
        )

        plt.figure(figsize=(10, 6))
        topic_engagement.plot(kind="bar", stacked=True)
        plt.title("Engagement by Tweet Topic")
        plt.xlabel("Topic")
        plt.ylabel("Average Engagement")
        plt.tight_layout()

        top_keywords = []
        for idx, topic in enumerate(lda.components_):
            keywords = [
                vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]
            ]
            top_keywords.append(f"Topic {idx}: {', '.join(keywords)}")

        return plt.gcf(), top_keywords
    except Exception as e:
        st.error(f"Error in engagement by topic plot: {e}")
        return None, []


# Function to perform sentiment analysis and plot results
def plot_sentiment_analysis(df_tweets):
    try:
        df_tweets["sentiment"] = df_tweets["text"].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )

        # Sentiment distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df_tweets["sentiment"], bins=30, kde=True, color="orange")
        plt.title("Tweet Sentiment Distribution")
        plt.xlabel("Sentiment Polarity")
        plt.tight_layout()
        fig_dist = plt.gcf()
        plt.close()

        # Sentiment vs engagement correlation
        sentiment_corr = df_tweets[["sentiment", "likes", "retweets"]].corr()
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            sentiment_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
        )
        plt.title("Sentiment vs Engagement Correlation")
        plt.tight_layout()
        fig_corr = plt.gcf()

        return fig_dist, fig_corr
    except Exception as e:
        st.error(f"Error in sentiment analysis plots: {e}")
        return None, None


# Load data
centrality_df, G, weekly_df, raw_df, df_tweets = load_data()

if (
    centrality_df is not None
    and G is not None
    and weekly_df is not None
    and raw_df is not None
    and df_tweets is not None
):
    weekly_df["community"] = weekly_df["community"].astype("category").cat.codes

    st.sidebar.header("Visualization Options")
    selected_viz = st.sidebar.multiselect(
        "Select Visualizations to Display",
        [
            "Influencer Network",
            "Centrality Comparison",
            "Followers vs Influencer Score",
            "Feature Importance",
            "Combined Influencer Predictions",
            "Average Engagement by Day",
            "Optimal Tweeting Times Table",
            "Optimal Tweeting Heatmap (FacetGrid)",
            "Optimal Tweeting Heatmaps (Individual)",
            "Engagement by Topic",
            "Sentiment Distribution",
            "Sentiment vs Engagement Correlation",
        ],
        default=[
            "Influencer Network",
            "Centrality Comparison",
            "Followers vs Influencer Score",
            "Feature Importance",
            "Combined Influencer Predictions",
            "Average Engagement by Day",
            "Optimal Tweeting Times Table",
            "Optimal Tweeting Heatmap (FacetGrid)",
            "Optimal Tweeting Heatmaps (Individual)",
            "Engagement by Topic",
            "Sentiment Distribution",
            "Sentiment vs Engagement Correlation",
        ],
    )

    if "Influencer Network" in selected_viz:
        st.subheader("Influencer Network - Centrality & Community")
        fig = plot_influencer_network(G, centrality_df)
        if fig:
            st.pyplot(fig)
        st.markdown(
            "This network graph shows influencers as nodes, sized by influencer score and colored by community, with edges representing following relationships."
        )

        st.subheader("Top 5 Influencers by Score")
        top_influencers = (
            centrality_df[["name", "screen_name", "followers_count", "influencer_score"]]
            .sort_values(by="influencer_score", ascending=False)
            .head(5)
        )
        st.dataframe(top_influencers)

        st.subheader("Top Influencers by Community")
        for comm_id in sorted(centrality_df["community"].unique()):
            st.write(f"Community {comm_id}")
            top_group = (
                centrality_df[centrality_df["community"] == comm_id][
                    ["screen_name", "followers_count", "influencer_score"]
                ]
                .sort_values("influencer_score", ascending=False)
                .head(3)
            )
            st.dataframe(top_group)

        st.write(f"Total communities detected: {len(set(centrality_df['community']))}")

    if "Centrality Comparison" in selected_viz:
        st.subheader("Centrality Comparison by Influencer")
        fig = plot_centrality_comparison(centrality_df)
        if fig:
            st.pyplot(fig)
        st.markdown(
            "This bar plot compares degree, betweenness, closeness, eigenvector, and PageRank centrality metrics across influencers."
        )

    if "Followers vs Influencer Score" in selected_viz:
        st.subheader("Followers vs Influencer Score")
        fig = plot_followers_vs_score(centrality_df)
        if fig:
            st.pyplot(fig)
        st.markdown(
            "This scatter plot shows the relationship between followers count and influencer score, with points colored by community and sized by PageRank."
        )

    if "Feature Importance" in selected_viz:
        st.subheader("Feature Importance for Engagement Prediction")
        fig = plot_feature_importance(weekly_df)
        if fig:
            st.pyplot(fig)
        st.markdown(
            "This bar plot shows the importance of features used in predicting engagement with a RandomForest model."
        )

    if "Combined Influencer Predictions" in selected_viz:
        st.subheader("Actual vs Predicted Engagement per Influencer")
        fig = plot_combined_influencer_predictions(weekly_df)
        if fig:
            st.pyplot(fig)
        st.markdown(
            "This multi-panel plot shows actual vs. predicted weekly engagement for each influencer over the last 4 weeks using LinearRegression."
        )

    if "Average Engagement by Day" in selected_viz:
        st.subheader("Average Engagement Rate by Day of Week")
        fig = plot_engagement_by_day(raw_df)
        if fig:
            st.pyplot(fig)
        st.markdown(
            "This bar plot shows the average engagement rate by day of the week based on tweet-level data."
        )

    if "Optimal Tweeting Times Table" in selected_viz:
        st.subheader("Optimal Tweeting Times per Influencer")
        optimal_times, best_times = get_optimal_tweeting_times(raw_df)
        if not optimal_times.empty:
            st.dataframe(optimal_times)
            st.markdown(
                "This table shows the day and hour with the highest average engagement for each influencer."
            )
        else:
            st.warning("No optimal tweeting times data available.")

    if "Optimal Tweeting Heatmap (FacetGrid)" in selected_viz:
        st.subheader("Optimal Tweeting Time by Influencer (FacetGrid)")
        if not best_times.empty:
            fig = plot_optimal_tweeting_heatmap_facet(best_times)
            if fig:
                st.pyplot(fig)
            st.markdown(
                "This FacetGrid heatmap shows average engagement by day of and hour for the top 20 influencers."
            )
        else:
            st.warning("No data available for optimal tweeting heatmap (FacetGrid).")

    if "Optimal Tweeting Heatmaps (Individual)" in selected_viz:
        st.subheader("Optimal Tweeting Heatmaps per Influencer")
        if not best_times.empty:
            fig = plot_optimal_tweeting_heatmaps_individual(best_times)
            if fig:
                st.pyplot(fig)
            st.markdown(
                "These heatmaps show average engagement by day and hour for each influencer individually."
            )
        else:
            st.warning("No data available for individual optimal tweeting heatmaps.")

    if "Engagement by Topic" in selected_viz:
        st.subheader("Engagement by Tweet Topic")
        fig, top_keywords = plot_engagement_by_topic(df_tweets)
        if fig:
            st.pyplot(fig)
            st.markdown(
                "This bar plot shows average likes and retweets per tweet topic."
            )
            if top_keywords:
                st.subheader("Top Keywords per Topic")
                for keyword in top_keywords:
                    st.write(keyword)
        else:
            st.warning("No data available for engagement by topic plot.")

    if (
        "Sentiment Distribution" in selected_viz
        or "Sentiment vs Engagement Correlation" in selected_viz
    ):
        st.subheader("Sentiment Analysis")
        fig_dist, fig_corr = plot_sentiment_analysis(df_tweets)
        if "Sentiment Distribution" in selected_viz and fig_dist:
            st.subheader("Tweet Sentiment Distribution")
            st.pyplot(fig_dist)
            st.markdown(
                "This histogram shows the distribution of tweet sentiment polarity using TextBlob."
            )
        if "Sentiment vs Engagement Correlation" in selected_viz and fig_corr:
            st.subheader("Sentiment vs Engagement Correlation")
            st.pyplot(fig_corr)
            st.markdown(
                "This heatmap shows the correlation between sentiment polarity and engagement metrics (likes, retweets)."
            )

    st.subheader("Challenges and Future Work")
    st.markdown(
        """
    ### Challenges
    1. **Incomplete or Noisy Data**
       - Many tweets were missing fields (e.g., full_text, engagement metrics), requiring data cleaning and imputation.
       - JSON structures varied across accounts, increasing parsing complexity.
    2. **Temporal Sparsity**
       - Some influencers lacked consistent weekly activity, making rolling window features less reliable.
    3. **Feature Correlation and Redundancy**
       - High correlation between features (e.g., retweets and likes) potentially introduced multicollinearity into models.
    4. **Centrality Metrics Misalignment**
       - The network-based centrality scores didn’t always correlate well with actual tweet engagement due to external influence factors (e.g., trending topics, external promotions).
    5. **Sentiment Polarity Limitations**
       - Using TextBlob for sentiment analysis was simplistic and failed to capture sarcasm, emojis, or context-sensitive sentiment.
    6. **Limited Explainability**

    ### Future Work
    1. **Incorporate Transformer-based NLP Models**
       - Replace TextBlob with more robust sentiment and semantic embeddings from models like BERT or RoBERTa for deeper tweet understanding.
    2. **Expand Social Graph Features**
       - Integrate dynamic social graphs and edge weights (e.g., frequency of mentions, retweets) to better reflect real-time influencer influence.
    3. **Real-time Prediction Dashboard**
       - Build a real-time dashboard that recommends optimal posting times and content themes per influencer using updated data streams.
    4. **Community Detection Impact Study**
       - Explore how community membership within the social graph (e.g., from Louvain or Leiden algorithms) influences engagement patterns.
    """
    )

else:
    st.warning(
        "No data available to display visualizations. Please check the data source and ensure all required files are present."
    )

# Footer
st.markdown("---")
st.markdown(
    f"Built with Streamlit | Data sourced from Twitter API and processed using pandas, seaborn, networkx, scikit-learn, and textblob | Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} +08"
)
