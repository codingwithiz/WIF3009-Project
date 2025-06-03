# app.py (final version)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import networkx as nx
from datetime import datetime
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

sns.set(style="whitegrid")
st.set_page_config(layout="wide")
st.title("📊 Influencer Analytics & SNA Dashboard")

# ----------------------------------
# Load Data
# ----------------------------------
@st.cache_data
def load_data():
    df_centrality = pd.read_csv("data/outputs/centrality/influencer_centrality.csv")
    df_graph = nx.read_gexf("data/processed/influencer_network.gexf")
    with open('data/raw/tweets_data.json', 'r') as f:
        tweets = json.load(f)
    return df_centrality, df_graph, tweets

# Load data
df_centrality, G, tweets = load_data()

# ----------------------------------
# Social Network Visualization
# ----------------------------------
st.header("🕸 Influencer Network Graph")
pos = nx.spring_layout(G, seed=42)
node_sizes = [(G.nodes[n]['followers_count'] / 1e6 + 1) * 200 for n in G.nodes]
node_colors = [G.nodes[n]['community'] if 'community' in G.nodes[n] else 0 for n in G.nodes]
fig_net, ax_net = plt.subplots(figsize=(12, 8))
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.Set2, ax=ax_net)
nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax_net)
nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]['screen_name'] for n in G.nodes}, font_size=9, ax=ax_net)
ax_net.set_title("Influencer Network: Size = Followers, Color = Community")
ax_net.axis("off")
st.pyplot(fig_net)

# ----------------------------------
# Centrality Comparison
# ----------------------------------
st.header("📊 Centrality Comparison by Influencer")
centrality_cols = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality', 'eigenvector_centrality', 'pagerank']
sorted_df = df_centrality.sort_values('influencer_score', ascending=False)
x = np.arange(len(sorted_df))
width = 0.15
fig_centrality, ax_centrality = plt.subplots(figsize=(14, 8))
for i, col in enumerate(centrality_cols):
    ax_centrality.bar(x + i*width, sorted_df[col], width, label=col)
ax_centrality.set_xticks(x + width*2)
ax_centrality.set_xticklabels(sorted_df['screen_name'], rotation=45, ha='right')
ax_centrality.set_title("Centrality Metrics per Influencer")
ax_centrality.legend()
st.pyplot(fig_centrality)

# ----------------------------------
# Followers vs Influencer Score
# ----------------------------------
st.header("📈 Followers vs Influencer Score")
fig_fs, ax_fs = plt.subplots(figsize=(10, 8))
sns.scatterplot(x='followers_count', y='influencer_score', hue='community', data=df_centrality, ax=ax_fs, palette='tab10')
for _, row in df_centrality.iterrows():
    ax_fs.annotate(row['screen_name'], (row['followers_count'], row['influencer_score']), fontsize=8)
ax_fs.set_title("Followers vs Influencer Score")
ax_fs.grid(True)
st.pyplot(fig_fs)

# ----------------------------------
# Load & Process Tweets
# ----------------------------------
st.header("🧠 NLP Analysis on Tweets")
rows = []
for user in tweets:
    instructions = user.get('tweets', {}).get('result', {}).get('timeline', {}).get('instructions', [])
    for instruction in instructions:
        if instruction.get('type') == 'TimelineAddEntries':
            entries = instruction.get('entries', [])
            for entry in entries:
                content = entry.get('content', {})
                item_content = content.get('itemContent', {})
                tweet_result = item_content.get('tweet_results', {}).get('result', {})
                legacy = tweet_result.get('legacy', {})
                text = legacy.get('full_text', "")
                timestamp = legacy.get('created_at', None)
                if text and timestamp:
                    try:
                        dt = datetime.strptime(timestamp, '%a %b %d %H:%M:%S %z %Y')
                        rows.append({
                            'text': text,
                            'hour': dt.hour,
                            'likes': legacy.get('favorite_count', 0),
                            'retweets': legacy.get('retweet_count', 0)
                        })
                    except:
                        continue

df_tweets = pd.DataFrame(rows)

# Avg likes by posting hour
st.subheader("🕒 Avg Likes by Posting Hour")
hourly_likes = df_tweets.groupby('hour')['likes'].mean()
best_hour = hourly_likes.idxmax()
fig_hour, ax_hour = plt.subplots()
hourly_likes.plot(kind='bar', color='skyblue', ax=ax_hour)
ax_hour.set_title("Average Likes by Hour")
st.pyplot(fig_hour)

# Sentiment distribution
st.subheader("💬 Tweet Sentiment Distribution")
df_tweets['sentiment'] = df_tweets['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
fig_sent, ax_sent = plt.subplots()
sns.histplot(df_tweets['sentiment'], bins=30, kde=True, color='orange', ax=ax_sent)
ax_sent.set_title("Sentiment Polarity Distribution")
st.pyplot(fig_sent)

# Sentiment vs engagement heatmap
st.subheader("📉 Sentiment vs Engagement Heatmap")
sentiment_corr = df_tweets[['sentiment', 'likes', 'retweets']].corr()
fig_heat, ax_heat = plt.subplots()
sns.heatmap(sentiment_corr, annot=True, cmap='coolwarm', ax=ax_heat)
ax_heat.set_title("Sentiment vs Engagement")
st.pyplot(fig_heat)

# ----------------------------------
# Future Prediction Chart
# ----------------------------------
st.header("🔮 Actual vs Predicted Influencer Score")
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Create synthetic time series
dates = pd.date_range(start='2023-01-01', periods=12, freq='W')
np.random.seed(42)
rows = []
for _, row in df_centrality.iterrows():
    for d in dates:
        rows.append({
            'username': row['screen_name'],
            'date': d,
            'followers_count': row['followers_count'],
            'engagement_rate': np.random.uniform(0.01, 0.2),
            'degree_centrality': row['degree_centrality'],
            'betweenness_centrality': row['betweenness_centrality'],
            'closeness_centrality': row['closeness_centrality'],
            'eigenvector_centrality': row['eigenvector_centrality'],
            'pagerank': row['pagerank']
        })
df_ts = pd.DataFrame(rows)
df_ts['influencer_score'] = df_ts['followers_count'] * df_ts['engagement_rate']
features = ['followers_count', 'engagement_rate', 'degree_centrality', 'betweenness_centrality', 'closeness_centrality', 'eigenvector_centrality', 'pagerank']
for f in features:
    df_ts[f'{f}_lag1'] = df_ts.groupby('username')[f].shift(1)
df_ts['influencer_score_next'] = df_ts.groupby('username')['influencer_score'].shift(-1)
df_model = df_ts.dropna()
X = df_model[[f + '_lag1' for f in features]]
y = df_model['influencer_score_next']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
fig_pred, ax_pred = plt.subplots()
ax_pred.scatter(y_test, y_pred, alpha=0.6)
ax_pred.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax_pred.set_xlabel("Actual")
ax_pred.set_ylabel("Predicted")
ax_pred.set_title("Actual vs Predicted Influencer Score")
st.pyplot(fig_pred)