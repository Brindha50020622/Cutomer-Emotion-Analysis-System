**Customer Emotion Analysis System**

The Customer Emotion Analysis System, implemented using deep learning
architecture aims to analyze customer feedback to identify emotional sentiment and key
topics, helping business understand customer satisfaction and engagement.This project
combines emotion classification and topic analysis to provide valuable insights from text
data.

Methodology:
Data Collection: Hugging Face’s GoEmotions dataset was used for training the
emotion classification model with 30,000 data points containing 28 different emotions.
Emotion Model: Used DistilBERT emotional classification. The model was fine-tuned
on the dataset to classify emotions under three activation levels(High, Medium and
Low).The intensity for each emotions(primary and secondary) was calculated using the
softmax function applied to the model’s logits, which converts raw output scores into
probabilities, representing the likelihood of each emotion class. The highest probability
corresponds to the primary emotion's confidence, while the second highest represents the
secondary emotion's confidence.
Topic Analysis: KeyBERT model was used. Bigrams (two-word phrases) are also
considered for extracting more meaningful insights. A predefined set of topics related to
customer feedback (such as "Delivery," "Quality," "Customer Service," etc.) is used to map
extracted keywords to specific topics.
Adorescore Calculation: Overall score was calculated by giving weight of 70% to
emotions(weights were assigned for three activation levels accordingly) and 30% to various
subtopics from topic analysis.For topics, all possible words were maintained as dictionary of
positive and negative words and scores were assigned accordingly for positive and negative
words.
Streamlit Web Application: The final system was integrated into a user-friendly
Streamlit Web Application.
Findings:

 The emotion classification model was able to identify the primary and
secondary emotions from the customer feedback. However, the model
achieved only 60% accuracy after training for five epochs due to time
constraints, which limited further training.
 Customer feedback was often polarized, with positive feedback revolving
around product features, while negative feedback focused on delivery delays or
customer service experiences.

Recommendations:

 Expand Vocabulary: As the system is extended, it is recommended to include
more industry-specific positive and negative words to further refine the
emotion and topic classification.
 Multiple Languages are not supported(did not train due to time constraints).

STEPS TO RUN:
1.Downlaod all the files(including all the saved models etc..)
2.customer_analysis_app.py is the integrated file of all three emotion analysis model,
topic analysis model and confidence score calculation using streamlit.
3.open customer_analysis_app.py and run using streamlit run
customer_analysis_app.py or python -m streamlit run customer_analysis_app.py
4.Streamlit application opens in browser
