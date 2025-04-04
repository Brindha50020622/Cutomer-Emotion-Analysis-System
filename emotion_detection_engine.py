import pandas as pd
import re
import nltk
import torch
from nltk.corpus import stopwords
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

nltk.download('stopwords')

data = pd.read_csv("goemotions_train_30000.csv")
columns_to_drop = ['id', 'author', 'subreddit', 'link_id', 'parent_id', 'created_utc', 'rater_id', 'example_very_unclear', 'labels']
data.drop(columns=columns_to_drop, inplace=True)

dominant_emotion = data.iloc[:, 1:].idxmax(axis=1)
data['emotion'] = dominant_emotion

# Clean text
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return ' '.join(word for word in text.split() if word not in stopwords.words('english'))

data['clean_text'] = data['text'].apply(clean_text)

# Encode Labels
emotions = data.columns[1:-2].tolist()
label2id = {emotion: idx for idx, emotion in enumerate(emotions)}
data['label'] = data['emotion'].map(label2id)

# Dataset Preparation
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding='max_length', max_length=max_len, return_tensors='pt')
        self.labels = torch.tensor(labels.tolist())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(data['clean_text'], data['label'], test_size=0.2, random_state=42)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_dataset = EmotionDataset(X_train, y_train, tokenizer)
val_dataset = EmotionDataset(X_val, y_val, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Load Model, Optimizer, and Loss
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(emotions))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

def train_epoch():
    model.train()
    total_loss, correct = 0, 0
    for batch, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(**{k: v.to(device) for k, v in batch.items()}, labels=labels.to(device))
        outputs.loss.backward()
        optimizer.step()
        total_loss += outputs.loss.item()
        correct += (outputs.logits.argmax(dim=1) == labels.to(device)).sum().item()
    return correct / len(train_dataset), total_loss / len(train_loader)

def eval_model():
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for batch, labels in val_loader:
            outputs = model(**{k: v.to(device) for k, v in batch.items()}, labels=labels.to(device))
            total_loss += outputs.loss.item()
            correct += (outputs.logits.argmax(dim=1) == labels.to(device)).sum().item()
    return correct / len(val_dataset), total_loss / len(val_loader)

for epoch in range(5):
    train_acc, train_loss = train_epoch()
    val_acc, val_loss = eval_model()
    print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f}, Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}, Loss: {val_loss:.4f}")

model.save_pretrained('emotion_classifier_distilbert')
tokenizer.save_pretrained('emotion_classifier_distilbert')

def predict_emotion(text):
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', max_length=64, truncation=True).to(device)
    with torch.no_grad():
        pred_idx = model(**inputs).logits.argmax(dim=1).item()
    return list(label2id.keys())[pred_idx]

print("Predicted Emotion:", predict_emotion("I am feeling so happy and excited about this project!"))

# Classification Report
def generate_report():
    true_labels, pred_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch, labels in val_loader:
            preds = model(**{k: v.to(device) for k, v in batch.items()}).logits.argmax(dim=1)
            true_labels.extend(labels.tolist())
            pred_labels.extend(preds.cpu().tolist())
    print(classification_report(true_labels, pred_labels, target_names=label2id.keys()))

generate_report()
