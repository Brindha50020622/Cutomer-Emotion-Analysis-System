import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from emotion_analysis.test import predict_emotions
from topic_analysis.topic_analysis import extract_topics_and_subtopics

# Activation level scores
ACTIVATION_SCORES = {
    "High Activation": 100,
    "Medium Activation": 70,
    "Low Activation": 50,
    "Unknown Activation": 0
}

# Subtopic Relevance Scores
SUBTOPIC_RELEVANCE_SCORES = {
    "High": 50,
    "Medium": 30,
    "Low": 10
}

POSITIVE_WORDS = {"fast", "good", "amazing", "excellent", "great", "loved", "fantastic", "quick", "satisfied", "perfect"}
NEGATIVE_WORDS = {"bad", "poor", "slow", "terrible", "horrible", "disappointed", "worst", "unhappy", "broken", "unfit"}


def calculate_emotion_score(emotion, weight):
    """Calculates emotion score based on confidence and activation level."""
    confidence = emotion["confidence"]
    activation = emotion["activation"]
    activation_score = ACTIVATION_SCORES.get(activation, 0)
    return round((confidence * weight) + activation_score, 2)


def assign_subtopic_relevance(subtopics, text):
    subtopic_relevance = {}
    words = set(text.lower().split())

    for topic, subtopic_list in subtopics.items():
        relevance_level = "Medium"  # Default level

        if words.intersection(POSITIVE_WORDS):
            relevance_level = "High"
        elif words.intersection(NEGATIVE_WORDS):
            relevance_level = "Low"

        subtopic_relevance[topic] = relevance_level

    return subtopic_relevance


def calculate_topic_relevance_score(main_topic, subtopics, subtopic_relevance, emotion_score):
    subtopic_scores = [SUBTOPIC_RELEVANCE_SCORES[subtopic_relevance.get(topic, "Low")] for topic in subtopics]
    avg_subtopic_score = round(sum(subtopic_scores) / len(subtopic_scores) if subtopic_scores else 0, 2)

    topic_score = len(subtopics) * 10  # Main topic score (each subtopic gives 10 points)
    total_topic_score = round(min(topic_score + avg_subtopic_score + emotion_score, 100), 2)  # Capped at 100

    return total_topic_score, avg_subtopic_score


def calculate_adorescore(text: str):
    #  Emotion Analysis
    emotion_results = predict_emotions(text)
    primary_emotion = emotion_results["Primary Emotion"]
    secondary_emotion = emotion_results["Secondary Emotion"]

    #  Topic Analysis
    topic_results = extract_topics_and_subtopics(text)
    main_topics = topic_results["topics"]["main"]
    subtopics = topic_results["topics"]["subtopics"]

    #  Assign Subtopic Relevance Levels
    subtopic_relevance = assign_subtopic_relevance(subtopics, text)

    #  Calculate scores for each main topic
    topic_scores = {}
    total_emotion_score = 0
    for main_topic in main_topics:
        primary_emotion_score = calculate_emotion_score(primary_emotion, weight=60)
        secondary_emotion_score = calculate_emotion_score(secondary_emotion, weight=40)
        total_emotion_score = round(primary_emotion_score + secondary_emotion_score, 2)

        total_topic_score, avg_subtopic_score = calculate_topic_relevance_score(main_topic, subtopics.get(main_topic, []), subtopic_relevance, total_emotion_score)

        topic_scores[main_topic] = total_topic_score

    # Calculate Overall Adorescore
    overall_score = 0
    topic_count = len(main_topics)
    if topic_count > 0:
        weighted_topic_score = round(sum(topic_scores.values()) / topic_count, 2)
        overall_score = round((0.7 * total_emotion_score) + (0.3 * weighted_topic_score), 2)

    return {
        "adorescore": {
            "overall": overall_score,
            "breakdown": {topic: round(score, 2) for topic, score in topic_scores.items()}
        }
    }


if __name__ == "__main__":
    sample_text = "I am happy with the product"
    results = calculate_adorescore(sample_text)

    print(f"\nAdorescore: {results['adorescore']['overall']}")
    print(f"Breakdown: {results['adorescore']['breakdown']}")
