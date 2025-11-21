import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import emoji

tokenizer = pickle.load(open("tokenizer.pickle", "rb"))
model = load_model("emoji_lstm_model.h5")

emoji_dict = {
    0: ":red_heart:",
    1: ":baseball:",
    2: ":grinning_face_with_big_eyes:",
    3: ":disappointed_face:",
    4: ":fork_and_knife_with_plate:"
}
# Max sequence length used during training
MAX_LEN = 20


def label_to_emoji(label):
    return emoji.emojize(emoji_dict[label])


# ----------------------------------
# Streamlit UI
# ----------------------------------
st.title("LSTM Emoji Predictor 😊🚀")
st.write("Enter text and the model predicts the best emoji.")


user_input = st.text_input("Enter your message:")


if st.button("Predict Emoji"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

        pred = model.predict(padded)
        label = np.argmax(pred)

        # st.success(f"Predicted Emoji: {emoji_dict.get(label, '❓')}")
        st.success(f"Predicted Emoji: {label_to_emoji(label)}")

        st.write("\n---\nMade with ❤️ using LSTM + Streamlit")
