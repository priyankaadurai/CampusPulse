import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv('/Users/oscarmandell/Downloads/archive (1)/roberta2022/aggies_2022_sampled.csv')
df['body'] = df['body'].fillna("").astype(str)
df.columns = df.columns.str.lower()
taxonomy = {
    'Housing': ['dorm', 'residence', 'housing', 'apartment', 'roommate', 'rent', 'lease', 'sublease'],
    'Academics': ['prof', 'professor', 'class', 'classes', 'major', 'study', 'finals', 'midterms', 'gpa', 'grade'],
    'Dining': ['dining hall', 'food', 'restaurants', 'groceries', 'grocery stores', 'meal plan'],
    'Safety': ['safety', 'crime', 'assault', 'SA', 'assault', 'robbery', 'robbed', 'danger', 'unsafe'],
    'Financial Aid': ['tuition', 'financial aid', 'scholarship', 'grant', 'student loans'],
    'Admissions': ['admissions', 'transfer', 'acceptance rate', 'SAT'],
    'Social Life': ['friends', 'clubs', 'bars', 'parties', 'social life', 'clubs', 'communities'],
    'Transportation': ['car', 'public transportation', 'transportation', 'buses', 'trains', 'biking', 'bicycle', 'scooter', 'walking']}

def assign_topic(text):
    text = str(text).lower()
    assigned_topics = []
    for topic, keywords in taxonomy.items():
        if any(word in text for word in keywords):
            assigned_topics.append(topic)
    return assigned_topics[0] if assigned_topics else 'Other'

df['topic'] = df['body'].apply(assign_topic)

df['net_sentiment'] = df['emo_pred_pos'] - df['emo_pred_neg']

topic_stats = df.groupby('topic').agg(
    volume = ('body', 'count'),
    avg_sentiment = ('net_sentiment', 'mean'), 
).reset_index()

def get_top_keywords(topic_df, n = 5):
    if len(topic_df) < 1:
        return ""
    vectorizer = TfidfVectorizer(stop_words = 'english', max_features = n)
    tfidf_matrix = vectorizer.fit_transform(topic_df['body'])
    return ", ".join(vectorizer.get_feature_names_out())

keywords_list = []
for topic in topic_stats['topic']:
    keywords_list.append(get_top_keywords(df[df['topic'] == topic]))
topic_stats['top_keywords'] = keywords_list

topic_stats_sorted = topic_stats.sort_values(by = 'avg_sentiment', ascending = False)
topic_stats_sorted.rename(columns = {'topic': 'TOPIC',
                                     'volume': 'VOLUME',
                                     'avg_sentiment': 'AVG SENTIMENT',
                                     'top_keywords': 'TOP KEYWORDS'}, inplace = True)
print(topic_stats_sorted[['TOPIC', 'VOLUME', 'AVG SENTIMENT', 'TOP KEYWORDS']].to_string(index = False))

heatmap_data = df.groupby('topic')[['emo_pred_pos', 'emo_pred_neu', 'emo_pred_neg']].mean()

plt.figure(figsize = (10,6))
sns.heatmap(heatmap_data, annot = True, cmap = 'RdYlGn', center = 0.5, fmt = ' .2f', cbar_kws = {'label': 'Average Score'})
plt.title('UC Davis 2022 Campus Life Sentiment Heatmap (Average Scores by Topic)')
plt.ylabel('Campus Topic')
plt.xlabel('Sentiment Category')
plt.tight_layout()
plt.savefig('UCD_2022_campus_sentiment_heatmap.png')


net_heatmap_data = df.groupby('topic')[['net_sentiment']].mean().sort_values(by = 'net_sentiment')

plt.clf()
plt.figure(figsize = (6, 8))
sns.heatmap(net_heatmap_data, annot = True, cmap = 'RdYlGn', center = 0, fmt = ' .2f')
plt.title('UC Davis 2022 Net Sentiment by Topic')
plt.ylabel('Campus Topic')
plt.xlabel('Net Score')
plt.tight_layout()
plt.savefig('UCD_2022_net_sentiment_heatmap.png')


lowest_sentiment = df.groupby('topic')['net_sentiment'].mean().sort_values().head(5).reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data = lowest_sentiment,
            x = 'net_sentiment',
            y = 'topic',
            hue = 'topic', 
            palette = 'Reds_r',
            legend = False)

plt.title('UC Davis 2022 Top 5 Campus "Pain Points" (Lowest Sentiment)', fontsize = 14)
plt.xlabel('Average Net Sentiment Score', fontsize = 12)
plt.ylabel('Campus Topic', fontsize = 12)

plt.axvline(0, color = 'black', linestyle = '--', linewidth = 1)

plt.xlim(-1, 1)
plt.tight_layout()
plt.savefig('UCD_2022_lowest_sentiment_barchart.png')

