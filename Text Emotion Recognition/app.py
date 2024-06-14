import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import joblib

pipe_svm = joblib.load(open("Model/text_emotion.pkl", "rb"))



def predict_emotions(docx):
    results = pipe_svm.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_svm.predict_proba([docx])
    return results


def main():
    st.title("Text Emotion Recognition")
    st.subheader("Detect Emotions in Text")
    with st.form(key="my_form"):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)
        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            st.write("Emotion: ", format(prediction))
            st.write("Confidence: ", format(np.max(probability)))

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_svm.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()
