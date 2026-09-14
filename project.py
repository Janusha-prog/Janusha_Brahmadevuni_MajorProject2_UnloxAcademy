import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# Load files

users = pd.read_csv("users_.csv")

tfidf_vectorizer = joblib.load(
    "tfidf_vectorizer.pkl"
)

user_weights = joblib.load(
    "user_adaptive_weights.pkl"
)

# Page settings

st.set_page_config(
    page_title="Profile Matching System",
    layout="wide"
)

st.title(" Intelligent Profile Matching System")
st.write(
    "Find compatible profiles using NLP, MBTI, "
    "location and personalized preferences."
)

st.divider()


# MBTI compatibility

def get_mbti_score(mbti1, mbti2):

    if mbti1 == mbti2:
        return 70

    pairs = {
        "INTJ": ["ENFP", "ENTP"],
        "INTP": ["ENTJ", "ENFJ"],
        "ENTJ": ["INTP", "INFP"],
        "ENTP": ["INTJ", "INFJ"],
        "INFJ": ["ENTP", "ESTP"],
        "INFP": ["ENTJ", "ENFJ"],
        "ENFJ": ["INFP", "INTP"],
        "ENFP": ["INTJ", "ISTJ"],
        "ISTJ": ["ENFP", "ESFP"],
        "ISFJ": ["ENTP", "ESTP"],
        "ESTJ": ["ISFP", "INFP"],
        "ESFJ": ["ISTP", "ISFP"],
        "ISTP": ["ESFJ", "ENFJ"],
        "ISFP": ["ESTJ", "ESFJ"],
        "ESTP": ["INFJ", "ISFJ"],
        "ESFP": ["ISTJ", "ISFJ"]
    }

    if mbti2 in pairs.get(mbti1, []):
        return 100

    return 50


# Select user

user_id = st.selectbox(
    "Select your User ID",
    users["user_id"]
)

user = users[
    users["user_id"] == user_id
].iloc[0]


# Display user profile

st.subheader("Your Profile")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Name", user["name"])
col2.metric("Profession", user["profession"])
col3.metric("MBTI", user["mbti"])
col4.metric("Location", user["location"])

st.write("**Professional Summary:**")
st.write(user["professional_summary"])

st.write("**About Me:**")
st.write(user["about_me"])

st.write("**Interests:**")
st.write(user["interests"])


# Find Top 5 matches

if st.button("🔍 Find My Top 5 Matches"):

    text_matrix = tfidf_vectorizer.transform(
        users["cleaned_profile_text"]
    )

    user_index = users.index[
        users["user_id"] == user_id
    ][0]

    weights = user_weights.get(
        user_id,
        {
            "text_weight": 0.50,
            "mbti_weight": 0.30,
            "location_weight": 0.20
        }
    )

    results = []

    for i, candidate in users.iterrows():

        if candidate["user_id"] == user_id:
            continue

        text_score = cosine_similarity(
            text_matrix[user_index],
            text_matrix[i]
        )[0][0] * 100

        mbti_score = get_mbti_score(
            user["mbti"],
            candidate["mbti"]
        )

        if user["location"] == candidate["location"]:
            location_score = 100
        else:
            location_score = 30

        final_score = (
            weights["text_weight"] * text_score
            + weights["mbti_weight"] * mbti_score
            + weights["location_weight"] * location_score
        )

        results.append({
            "User ID": candidate["user_id"],
            "Name": candidate["name"],
            "Profession": candidate["profession"],
            "Location": candidate["location"],
            "MBTI": candidate["mbti"],
            "Score": final_score,
            "Text": text_score,
            "MBTI Score": mbti_score,
            "Location Score": location_score
        })

    results = pd.DataFrame(results)

    results = results.sort_values(
        "Score",
        ascending=False
    )

    # Remember Top 5 results
    st.session_state["top5"] = results.head(5)


# Display Top 5 if available

if "top5" in st.session_state:
    top5 = st.session_state["top5"]

    # Get personalized weights again after Streamlit reruns
    weights = user_weights.get(
        user_id,
        {
            "text_weight": 0.50,
            "mbti_weight": 0.30,
            "location_weight": 0.20
        }
    )

    st.divider()

    st.subheader("🎯 Your Top 5 Matches")

    for rank, (_, match) in enumerate(
        top5.iterrows(),
        start=1
    ):

        st.markdown(
            f"### #{rank} — {match['Name']}"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Compatibility",
            f"{match['Score']:.2f}%"
        )

        c2.write(
            f"**Profession**\n\n"
            f"{match['Profession']}"
        )

        c3.write(
            f"**MBTI**\n\n"
            f"{match['MBTI']}"
        )

        c4.write(
            f"**Location**\n\n"
            f"{match['Location']}"
        )

        with st.expander(
            "Why was this person recommended?"
        ):

            st.write(
                f"Text Similarity: "
                f"{match['Text']:.2f}%"
            )

            st.write(
                f"MBTI Compatibility: "
                f"{match['MBTI Score']:.0f}%"
            )

            st.write(
                f"Location Compatibility: "
                f"{match['Location Score']:.0f}%"
            )

            st.write("**Personalized Weights:**")

            st.write(
                f"Text: "
                f"{weights['text_weight'] * 100:.1f}%"
            )

            st.write(
                f"MBTI: "
                f"{weights['mbti_weight'] * 100:.1f}%"
            )

            st.write(
                f"Location: "
                f"{weights['location_weight'] * 100:.1f}%"
            )

        # Feedback buttons

        col_a, col_b = st.columns(2)

        with col_a:

            if st.button(
                "Accept",
                key=f"accept_{match['User ID']}"
            ):

                new_feedback = pd.DataFrame([{
                    "user_id": user_id,
                    "matched_user_id": match["User ID"],
                    "action": 1,
                    "timestamp": pd.Timestamp.now()
                }])

                new_feedback.to_csv(
                    "feedback_.csv",
                    mode="a",
                    header=False,
                    index=False
                )

                st.success(
                    "Feedback recorded: Accept"
                )

        with col_b:

            if st.button(
                "Reject",
                key=f"reject_{match['User ID']}"
            ):

                new_feedback = pd.DataFrame([{
                    "user_id": user_id,
                    "matched_user_id": match["User ID"],
                    "action": 0,
                    "timestamp": pd.Timestamp.now()
                }])

                new_feedback.to_csv(
                    "feedback_.csv",
                    mode="a",
                    header=False,
                    index=False
                )

                st.success(
                    "Feedback recorded: Reject"
                )

        st.divider()