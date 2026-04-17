import pandas as pd
import numpy as np


df = pd.read_csv('/Users/oscarmandell/Downloads/archive (1)/roberta2022/yale_2022_sampled.csv')
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
    for topic, keywords in taxonomy.items():
        if any(word in text for word in keywords):
            return topic
    return 'Miscellaneous'

df['topic'] = df['body'].apply(assign_topic)

df['net_sentiment'] = df['emo_pred_pos'] - df['emo_pred_neg']

topic_stats = df.groupby('topic').agg(
    volume = ('body', 'count'),
    avg_sentiment = ('net_sentiment', 'mean')
).sort_values(by = 'avg_sentiment', ascending = False)

print(topic_stats)





