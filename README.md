# Janusha_Brahmadevuni_MajorProject2_UnloxAcademy

# Intelligent Hybrid Profile Matching System

## Project Overview

This project is a profile matching and recommendation system that helps users find suitable profiles based on different factors.

The system mainly considers the professional information, personality type, location, interests, and previous user feedback. Natural Language Processing is used to compare the text information provided in user profiles.

The project combines these different factors to calculate a compatibility score and recommend the Top 5 matching profiles.

## Objectives

- To process user profile information using NLP.
- To compare profiles based on their professional and personal information.
- To calculate text similarity using TF-IDF and cosine similarity.
- To include MBTI personality compatibility.
- To consider location while finding matches.
- To combine different factors into one compatibility score.
- To learn from user Accept and Reject feedback.
- To generate personalized recommendations.
- To provide the Top 5 matching profiles through a simple web application.

## Dataset

The project uses two main datasets.

### users_.csv

This file contains 100 synthetic user profiles.
The dataset includes the following information:

- User ID
- Name
- Age
- Location
- Profession
- Experience
- Professional Summary
- About Me
- MBTI Personality Type
- Interests

### feedback_.csv

This file contains user feedback for profile recommendations.
The feedback is represented using:

- 1 for Accept
- 0 for Reject

The dataset contains 800 feedback interactions.
The datasets were generated using Python and are synthetic. They were created only for educational and project purposes.

## Methodology

The project follows a hybrid profile matching approach.

First, the profile information is collected and prepared for processing. The text fields such as Professional Summary, About Me, and Interests are combined and processed using basic NLP techniques.

TF-IDF is then used to convert the text into numerical features. Cosine similarity is used to measure how similar two profiles are.

MBTI personality types are also compared using predefined compatibility rules. Location is considered as another factor in the matching process.

These factors are combined to calculate the final compatibility score.

User feedback is then used to understand which types of matches are more suitable for a particular user. Personalized weights are calculated based on the available feedback.

Finally, the system uses these scores to recommend the Top 5 profiles.

## Natural Language Processing

Basic NLP techniques are used for processing the profile text.
The preprocessing includes:

- Converting text to lowercase
- Removing punctuation
- Removing stopwords
- Lemmatization

After preprocessing, TF-IDF Vectorization is used to represent the text numerically.
Cosine similarity is used to calculate the similarity between two profile texts.

## Compatibility Score

The compatibility score is calculated using three main factors:

- Text similarity
- MBTI compatibility
- Location compatibility

The initial weights used in the system are:

- Text similarity: 50%
- MBTI compatibility: 30%
- Location compatibility: 20%

The final score is converted into a value between 0 and 100.
The system also uses personalized weights learned from user feedback.

## Machine Learning

Logistic Regression is used to learn from the Accept and Reject feedback.
The original Logistic Regression model achieved an accuracy of 70%.
A balanced Logistic Regression model was also tested because the feedback dataset contains more Reject interactions than Accept interactions.
The balanced model achieved an accuracy of 68.75%.
Although the overall accuracy was slightly lower, the Accept-class F1-score improved from 0.43 to 0.62.
This shows that the balanced model was better at identifying accepted matches.

## Personalized Feedback

The system also includes a feedback-based learning component.
Users can Accept or Reject recommended profiles.

This feedback is used to calculate personalized weights for each user. These weights determine how important text similarity, MBTI compatibility, and location are for that particular user.
For example, one user may give more importance to professional similarity, while another user may give more importance to personality compatibility.

The current synthetic test data did not show an improvement in holdout classification accuracy after personalization. However, the personalized weighting mechanism demonstrates how the system can adapt to individual preferences.

## Recommendation System

The recommendation system allows a user to select their profile and find suitable profiles.
For every possible match, the system calculates:

- Text similarity score
- MBTI compatibility score
- Location score
- Personalized compatibility score

The profiles are then sorted based on their final compatibility score.
The Top 5 profiles are displayed to the user.

## Explainability

The system also provides an explanation for each recommendation.
The user can see the factors that contributed to the compatibility score, such as:

- Text similarity
- MBTI compatibility
- Location compatibility
- Personalized weights

This makes the recommendation easier to understand instead of showing only a final score.

## Streamlit Application

A Streamlit web application was created for the project.
The application allows the user to:

1. Select a profile.
2. View the selected user's information.
3. Find the Top 5 compatible profiles.
4. View the compatibility score of each profile.
5. See the main factors behind the recommendation.
6. Give Accept or Reject feedback.

## Technologies Used:

Python
Pandas
NumPy
Scikit-learn
NLTK
Matplotlib
Seaborn
Streamlit
Joblib

## Model Evaluation:

The project includes model evaluation using accuracy, precision, recall, F1-score, and confusion matrix.
The original Logistic Regression model achieved 70% accuracy.
The balanced Logistic Regression model achieved 68.75% accuracy.
The balanced model performed better for identifying accepted matches, which is important because the recommendation system needs to identify profiles that users are likely to accept.

## Error Analysis:

Error analysis was performed to understand the incorrect predictions made by the final model.
The final evaluation contained 50 incorrect predictions out of 160 test samples.
Both false positives and false negatives were present.
False positives occur when the model predicts that a match will be accepted but the actual feedback is Reject.
False negatives occur when the model predicts Reject even though the actual feedback is Accept.
These errors are expected because profile compatibility depends on multiple factors and the dataset used in this project is synthetic.

## Limitations:

The dataset used in this project is synthetic.
The MBTI compatibility rules are predefined project-specific rules.
The system uses TF-IDF instead of advanced transformer-based NLP models.
The amount of feedback data is limited.
The current system uses a CSV file for storing feedback.
The personalized learning approach can be improved with more real user interactions.

## Future Scope:

The project can be improved in several ways in the future:
Use larger real-world datasets.
Use advanced NLP models for better text understanding.
Add more user preferences and behavioral information.
Improve personalized recommendation techniques.
Use a database instead of CSV files.
Add user authentication.
Store feedback in a permanent database.
Improve the recommendation model using more real user feedback.
Develop the system further into a complete production-level application.

## Conclusion:

This project demonstrates a hybrid profile matching system that combines Natural Language Processing, TF-IDF text similarity, MBTI compatibility, location matching, and feedback-based personalization.
The system can calculate compatibility scores and recommend the Top 5 profiles to a user.
The feedback component also provides a basic way for the system to learn individual user preferences.

Overall, the project shows how different machine learning and NLP techniques can be combined to build a practical profile recommendation system.
