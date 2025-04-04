import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re
import torch.nn.functional as F  

model_path = 'emotion_classifier_distilbert'
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Emotion Mapping
emotions = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
    'remorse', 'sadness', 'surprise', 'neutral'
]
label2id = {emotion: idx for idx, emotion in enumerate(emotions)}
id2label = {idx: emotion for emotion, idx in label2id.items()}

# Activation Levels
high_activation = {
    'ecstasy', 'rage', 'vigilance', 'oathing', 'grief', 'amazement', 'terror', 'admiration', 'vigilance','excitement','love'
}
medium_activation = {
    'anger', 'joy', 'anticipation', 'trust', 'fear', 'surprise', 'sadness', 'disgust','neutral'
}
low_activation = {
    'serenity', 'interest', 'acceptance', 'apprehension', 'distraction', 'pensiveness', 'boredom', 'annoyance','disappointment'
}

def get_activation(emotion):
    if emotion in high_activation:
        return "High Activation"
    elif emotion in medium_activation:
        return "Medium Activation"
    elif emotion in low_activation:
        return "Low Activation"
    else:
        return "Unknown Activation"

# Prediction Function with Primary & Secondary Emotions
def predict_emotions(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Clean text

    inputs = tokenizer(text, return_tensors='pt', padding='max_length', max_length=64, truncation=True).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze()

        top2_indices = probs.argsort(descending=True)[:2]

        primary_idx, secondary_idx = top2_indices.tolist()
        primary_emotion = id2label[primary_idx]
        secondary_emotion = id2label[secondary_idx]

        # Confidence scores
        primary_confidence = round(probs[primary_idx].item() , 2)
        secondary_confidence = round(probs[secondary_idx].item() , 2)

        # Activation levels
        primary_activation = get_activation(primary_emotion)
        secondary_activation = get_activation(secondary_emotion)

    return {
        "Primary Emotion": {"emotion": primary_emotion, "confidence": primary_confidence, "activation": primary_activation},
        "Secondary Emotion": {"emotion": secondary_emotion, "confidence": secondary_confidence, "activation": secondary_activation}
    }

'''# Step 4: Test Inputs
test_inputs = [
    "I am happy with the product"       

    
]

for text in test_inputs:
    result = predict_emotions(text)
    print(f"Input: {text}")
    print(f"Primary Emotion: {result['Primary Emotion']['emotion']} | Confidence: {result['Primary Emotion']['confidence']} | Activation: {result['Primary Emotion']['activation']}")
    print(f"Secondary Emotion: {result['Secondary Emotion']['emotion']} | Confidence: {result['Secondary Emotion']['confidence']} | Activation: {result['Secondary Emotion']['activation']}\n")
'''