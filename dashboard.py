import streamlit as st
import pandas as pd
import time

st.title("AI Surveillance Dashboard")

while True:
    df = pd.read_csv("outputs/log.csv")

    if len(df) > 0:
        st.metric("People Count", int(df.iloc[-1]["person_count"]))
        st.line_chart(df["person_count"])

    time.sleep(1)