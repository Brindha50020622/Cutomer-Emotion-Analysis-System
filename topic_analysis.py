from keybert import KeyBERT
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize KeyBERT for keyword extraction
kw_model = KeyBERT()

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]

    return words


def extract_keywords(text: str, top_n: int = 5):
    # Use KeyBERT for keyword extraction with bigrams
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    return [kw[0] for kw in keywords]  


TOPIC_MAPPING = {
    "Delivery": ["deliver", "fast", "slow", "on time", "late", "ship", "shipping", "arrive", "delay"],
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

def map_keywords_to_topics(keywords: list, tokenized_text: list):
    main_topics = []
    subtopics_dict = {}

    for topic, topic_keywords in TOPIC_MAPPING.items():
        # Check which keywords from the input match the topic's keywords
        matched_keywords = [kw for kw in topic_keywords if kw in tokenized_text]
        if matched_keywords:
            main_topics.append(topic)
            subtopics_dict[topic] = matched_keywords  

    # If no topics matched, default to keywords as main topics
    if not main_topics:
        main_topics = keywords[:3]
        subtopics_dict = {main_topics[0]: keywords[1:4]} if len(keywords) > 1 else {main_topics[0]: []}

    return {
        "topics": {
            "main": main_topics,
            "subtopics": subtopics_dict
        }
    }


def extract_topics_and_subtopics(text: str):
    tokenized_text = preprocess_text(text)
    keywords = extract_keywords(text, top_n=5)
    return map_keywords_to_topics(keywords, tokenized_text)

