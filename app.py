import streamlit as st
from transformers import pipeline

# Load the emotion classifier pipeline
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Emoji map for emotions
emoji_map = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲",
    "neutral": "😐",
    "disgust": "🤢",
    "guilt": "😔",
    "gratitude": "🙏",
    "pride": "😌",
    "confusion": "😕",
    "realization": "💡",
    "optimism": "🌈",
    "admiration": "👏"
}

# Session state initialization
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat_active' not in st.session_state:
    st.session_state.chat_active = False

st.title("🧠 Emotional Chat Classifier with Emojis 🎭")

# Buttons to control chat session
col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 Start Chat"):
        st.session_state.chat_active = True
        st.session_state.chat_history = []

with col2:
    if st.button("🛑 End Chat"):
        st.session_state.chat_active = False
        st.session_state.chat_history = []

# Chat UI
if st.session_state.chat_active:
    user_input = st.text_input("You:", key="chat_input")

    if user_input:
        # Get emotion prediction
        result = classifier(user_input)[0]
        emotion = result['label'].lower()
        score = round(result['score'], 2)
        emoji = emoji_map.get(emotion, "❓")

        # Store in history
        st.session_state.chat_history.append({
            "text": user_input,
            "emotion": emotion,
            "score": score,
            "emoji": emoji
        })

        # Refresh to clear input
        st.experimental_rerun()

    # Show chat history
    st.markdown("---")
    for entry in st.session_state.chat_history:
        st.markdown(f"**You**: {entry['text']}")
        st.markdown(f"{entry['emoji']} **Emotion**: `{entry['emotion'].capitalize()}` (Confidence: `{entry['score']}`)")
        st.markdown("---")
else:
    st.info("Click **Start Chat** to begin the conversation!")
