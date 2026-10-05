import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(
    page_title="Internship Recommendation System",
    page_icon="🎓",
    layout="wide"
)
# Sidebar
st.sidebar.title("🎓 Internship System")
st.sidebar.write("AI-Powered Internship Recommendation")
st.sidebar.markdown("---")
st.sidebar.write("### Features")
st.sidebar.write("🔹 Student Profile")
st.sidebar.write("🔹 Internship Search")
st.sidebar.write("🔹 AI Recommendations")
st.sidebar.write("🔹 Match Score")
st.title("🎓 AI-Powered Internship Recommendation System")
st.write(
    "This system recommends suitable internships "
    "based on a student's skills and interests."
)
# Load internship data
df = pd.read_csv("internships.csv")
# Student details
st.header("👨‍🎓 Enter Your Details")
col1, col2 = st.columns(2)
with col1:
    name = st.text_input("Enter your name")
    branch = st.selectbox(
        "Select your branch",
        ["CSE", "IT", "ECE", "EEE", "Mechanical", "Civil"]
    )
with col2:
    skills = st.text_input(
        "Enter your skills",
        placeholder="Example: Python, SQL, Pandas"
    )
    interest = st.selectbox(
        "Select your interest",
        [
            "Python",
            "Data Science",
            "Machine Learning",
            "Web Development",
            "Java",
            "Artificial Intelligence"
        ]
    )
location = st.selectbox(
    "Preferred Location",
    ["Any", "Hyderabad", "Bangalore", "Chennai"]
)
# Recommendation button
if st.button("🔍 Recommend Internships"):
    if skills.strip() == "":
        st.warning("Please enter your skills.")
        st.stop()
    # Combine student skills and interest
    student_profile = (
        skills + " " + interest + " " + branch
    )    
    # Combine internship skills and title
    internship_text = (
        df["Skills"] + " " + df["Title"]
    )
    # Convert text into numbers
    vectorizer = TfidfVectorizer()
    internship_vectors = vectorizer.fit_transform(
        internship_text
    )

    student_vector = vectorizer.transform(
        [student_profile]
    )

    # Calculate similarity
    similarity_scores = cosine_similarity(
        student_vector,
        internship_vectors
    )[0]

    # Convert similarity into percentage
    df["Match Score"] = similarity_scores * 100

    # Filter by location
    if location != "Any":
        df = df[
            df["Location"].str.lower()
            == location.lower()
        ]

    # Sort from highest match to lowest
    recommendations = df.sort_values(
        by="Match Score",
        ascending=False
    ).head(5)
    if recommendations.empty:
        st.warning("No internships found for the selected location.")
        st.stop()
    # Show recommendation table
    st.subheader("📋 Top Internship Matches")
    st.dataframe(
        recommendations[
            [
                "Title",
                "Company",
                "Location",
                "Skills",
                "Stipend",
                "Duration",
                "Match Score"
            ]
        ],
        use_container_width=True
    )
    # Show recommendations
    st.header("🌟 Recommended Internships")
    st.info("These internships are selected based on your skills, interest, and preferred location.")
    st.write(
        f"Hello **{name}**! Here are your best matches:"
    )
    for _, row in recommendations.iterrows():
        st.markdown("---")

        st.subheader(
            f"💼 {row['Title']}"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write("🏢 **Company:**", row["Company"])
            st.write("📍 **Location:**", row["Location"])
            st.write("🛠️ **Skills:**", row["Skills"])

        with col2:
            st.write("💰 **Stipend:** ₹", row["Stipend"])
            st.write("⏳ **Duration:**", row["Duration"])
            st.success(
                f"⭐ Match Score: {row['Match Score']:.2f}%"
            )
            st.progress(
                min(int(row["Match Score"]), 100)
            )
# --------------------------------------------------
# ABOUT THE PROJECT
# --------------------------------------------------
st.markdown("---")
with st.expander("📘 About This Project"):
    st.write(
        "The AI-Powered Internship Recommendation System "
        "helps students find suitable internships based on "
        "their skills, interests, branch, and preferred location."
    )

    st.write(
        "The system uses TF-IDF and cosine similarity "
        "to calculate the matching score between the "
        "student profile and available internships."
    )

    st.write(
        "Technologies used: Python, Streamlit, Pandas, "
        "NumPy, and Scikit-learn."
    )
# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption(
        "🎓 AI-Powered Internship Recommendation System | "
        "Python + Streamlit + Machine Learning"
)
# --------------------------------------------------
# RESET BUTTON
# --------------------------------------------------
if st.button("🔄 Reset"):
    st.rerun()