import sys
import os
sys.path.append(os.path.abspath(".."))

import streamlit as st
from preprocessing.eeg_data import load_and_preprocess, extract_epochs_and_labels
from models.csp import run_csp_svm
from models.meta_model import build_meta_dataset, train_meta_model, predict_correctness
from llm.feedback_generator import generate_feedback

st.set_page_config(page_title=" BCI Feedback Co-Pilot", layout="centered")

st.title(" AI-Augmented BCI Simulator")
st.markdown("Get real-time feedback on your brain's motor imagery performance.")

if st.button(" Run New EEG Trial"):
    with st.spinner("Loading EEG + Training model..."):
        raw = load_and_preprocess()
        X, y = extract_epochs_and_labels(raw)
        acc, model = run_csp_svm(X, y)

        csp = model.named_steps['csp']
        svm = model.named_steps['svm']
        X_csp = csp.transform(X)

        meta_X, meta_y = build_meta_dataset(svm, X_csp, y)
        meta_model = train_meta_model(meta_X, meta_y)

        trial_idx = 0
        prediction = svm.predict([X_csp[trial_idx]])[0]
        confidence = svm.decision_function([X_csp[trial_idx]])[0]
        reliability = predict_correctness(meta_model, confidence)
        history = ['low'] if reliability < 0.5 else ['high']

        feedback = generate_feedback(prediction, confidence, history)

    st.success("Trial complete!")
    st.markdown(f"**Prediction**: {'Right hand' if prediction else 'Left hand'}")
    st.markdown(f"**Confidence**: `{confidence:.2f}`")
    st.markdown(f"**Reliability** (P[correct]): `{reliability:.2f}`")
    st.info(f" GPT Feedback:\n\n{feedback}")


