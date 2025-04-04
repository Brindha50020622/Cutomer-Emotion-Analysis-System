import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("emotion_dataset/goemotions_train.csv")

emotion_columns = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
                   'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
                   'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
                   'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 
                   'remorse', 'sadness', 'surprise', 'neutral']

df['labels'] = df[emotion_columns].apply(lambda row: [emotion for emotion, val in row.items() if val == 1], axis=1)

df = df[df['labels'].map(len) > 0]

# 5. Perform stratified sampling based on the first emotion in the 'labels' list
df_sampled, _ = train_test_split(df, train_size=30000, stratify=df['labels'].apply(lambda x: x[0]), random_state=42)

# 6. Save the sampled dataset
df_sampled.to_csv("goemotions_train_30000.csv", index=False)

print("Reduced dataset with labels saved as 'goemotions_train_30000.csv'")
