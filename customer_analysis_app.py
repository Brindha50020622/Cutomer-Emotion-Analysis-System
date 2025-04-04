import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re
import torch.nn.functional as F
from keybert import KeyBERT
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import json

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load Emotion Model and Tokenizer
model_path = 'emotion_classifier_distilbert'
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Emotion Analysis Setup
emotions = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
            'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
            'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
            'remorse', 'sadness', 'surprise', 'neutral']
id2label = {idx: emotion for idx, emotion in enumerate(emotions)}

activation_levels = {
    "High Activation": { 'ecstasy', 'rage', 'vigilance', 'oathing', 'grief', 'amazement', 'terror', 'admiration', 'vigilance','excitement','love'},
    "Medium Activation": { 'anger', 'joy', 'anticipation', 'trust', 'fear', 'surprise', 'sadness', 'disgust','neutral'},
    "Low Activation": {'serenity', 'interest', 'acceptance', 'apprehension', 'distraction', 'pensiveness', 'boredom', 'annoyance','disappointment'}
}

def get_activation(emotion):
    for level, emotions_set in activation_levels.items():
        if emotion in emotions_set:
            return level
    return "Unknown"

def predict_emotions(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', max_length=64, truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze()
        top2_indices = probs.argsort(descending=True)[:2].tolist()
    
    primary_emotion = id2label[top2_indices[0]]
    secondary_emotion = id2label[top2_indices[1]]

    return {
        "primary": {"emotion": primary_emotion, "activation": get_activation(primary_emotion), "intensity": round(probs[top2_indices[0]].item(), 2)},
        "secondary": {"emotion": secondary_emotion, "activation": get_activation(secondary_emotion), "intensity": round(probs[top2_indices[1]].item(), 2)}
    }


kw_model = KeyBERT()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = [lemmatizer.lemmatize(w) for w in word_tokenize(text) if w not in set(stopwords.words('english'))]
    return words

def extract_topics(text, top_n=5):
    keywords = [kw[0] for kw in kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)]
    topic_mapping = {
       "Delivery": ["delivery", "fast", "slow", "on time", "late", "ship", "shipping", "arrive", "delay"],
    "Quality": ["quality", "poor", "good", "cheap", "durable", "material", "fabric", "uncomfortable", "fit", "comfort"],
    "Customer Service": ["service", "helpful", "friendly", "rude", "accommodating", "support", "staff", "assistance"],
    "Price": ["price", "expensive", "cheap", "reasonable", "affordable", "cost", "value", "budget"],
    "Return Policy": ["return", "policy", "complicated", "strict", "easy", "refund", "exchange", "warranty"],
    "Product Features": ["feature", "design", "size", "color", "style", "performance", "functionality", "ease of use", "usability", "innovation"],
    "Packaging": ["packaging", "box", "wrap", "damage", "secure", "fragile", "presentation", "unboxing"],
    "Shipping and Handling": ["shipping", "handling", "delivery", "package", "carrier", "tracking", "damaged", "lost", "delay"],
    "Product Durability": ["durable", "long-lasting", "break", "tear", "wear", "sturdy", "fragile", "lasting"],
    "Product Performance": ["performance", "speed", "efficiency", "accuracy", "reliability", "effectiveness", "output", "result"],
    "Customer Experience": ["experience", "satisfaction", "disappointment", "enjoy", "love", "hate", "recommend", "regret"],
    "Product Availability": ["availability", "stock", "out of stock", "backorder", "wait", "pre-order", "limited"],
    "Product Instructions": ["instructions", "manual", "guide", "confusing", "clear", "helpful", "setup", "installation"],
    "Product Safety": ["safe", "dangerous", "hazard", "risk", "child-safe", "certified", "warranty", "guarantee"],
    "Product Aesthetics": ["look", "appearance", "design", "style", "color", "shape", "attractive", "ugly"],
    "Product Compatibility": ["compatible", "fit", "work with", "integration", "adaptable", "universal"],
    "Product Value for Money": ["value", "worth", "overpriced", "bargain", "deal", "investment", "rip-off"],
    "Product Maintenance": ["maintenance", "clean", "care", "repair", "service", "upkeep", "durability"],
    "Product Brand": ["brand", "reputation", "trust", "quality", "popular", "reliable", "premium"],
    "Product Comparison": ["compare", "better than", "worse than", "alternative", "competitor", "similar"],
    "Product Customization": ["custom", "personalized", "tailored", "options", "choices", "configurable"],
    "Product Sustainability": ["eco-friendly", "sustainable", "environment", "green", "recyclable", "carbon footprint"],
    "Product Technology": ["technology", "advanced", "innovative", "cutting-edge", "outdated", "modern"],
    "Product Comfort": ["comfort", "comfortable", "ergonomic", "pain", "ease", "relaxing"],
    "Product Noise Level": ["noise", "quiet", "loud", "silent", "disturbing", "sound"]
    }
    tokenized = preprocess_text(text)

    main_topics, subtopics = [], {}
    for topic, words in topic_mapping.items():
        matched = [w for w in words if w in tokenized]
        if matched:
            main_topics.append(topic)
            subtopics[topic] = matched

    if not main_topics:
        main_topics = keywords[:3]
        subtopics = {main_topics[0]: keywords[1:]} if len(keywords) > 1 else {main_topics[0]: []}

    return {"main": main_topics, "subtopics": subtopics}

ACTIVATION_SCORES = {"High Activation": 100, "Medium Activation": 70, "Low Activation": 50, "Unknown": 0}
SUBTOPIC_SCORES = {"High": 50, "Medium": 30, "Low": 10}
POSITIVE = {"fast", "good", "excellent", "quick", "loved", "love", "happy", "satisfied", "amazing", "great", "positive", "awesome", "fantastic", "wonderful", "outstanding"}
NEGATIVE = {"bad", "poor", "slow", "cheap", "tight", "uncomfortable", "disappointing", "horrible", "terrible", "negative", "awful", "worst", "annoying"}

def adorescore(text, emotions, topics):
    # Calculate activation score
    activation_score = (ACTIVATION_SCORES.get(emotions['primary']['activation'], 0) + ACTIVATION_SCORES.get(emotions['secondary']['activation'], 0)) / 2
    
    # Assign subtopic scores based on positive/negative words
    def calculate_subtopic_score(subtopic):
        score = 0
        words_in_subtopic = subtopic.split()  
        for word in words_in_subtopic:
            if word.lower() in POSITIVE:
                score += 5  # Assign score for positive words
            elif word.lower() in NEGATIVE:
                score -= 5  # Assign penalty for negative words
        
        # Ensure the subtopic score stays within the range of 0-100
        return min(max(score, 0), 100)
    
    subtopic_scores = {t: calculate_subtopic_score(" ".join(subs)) for t, subs in topics['subtopics'].items()}
    
    breakdown = {topic: min(score + activation_score, 100) for topic, score in subtopic_scores.items()}
    
    # Calculate the overall score
    if breakdown:
        overall = round((0.7 * activation_score) + (0.3 * (sum(breakdown.values()) / len(breakdown))), 2)
    else:
        overall = round(activation_score, 2)
    
    return {"overall": min(overall, 100), "breakdown": breakdown}



st.title("Customer Emotion Analysis System")
input_text = st.text_area("Enter customer feedback:")

if st.button("Analyze") and input_text:
    emotions = predict_emotions(input_text)
    topics = extract_topics(input_text)
    score = adorescore(input_text, emotions, topics)

    result = {
        "emotions": emotions,
        "topics": topics,
        "adorescore": score
    }

    st.subheader("Output (JSON Format)")
    st.json(result)
