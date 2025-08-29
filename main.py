# python -m streamlit run main.py

#st.write("hello world")
#x= st.text_input("Meow?")
#st.write("## Header")
#st.write("``` code block ```")
#data = pd.read_csv("keeper_combined.csv")
#st.write(data)
#st.bar_chart(data.League)
#st.line_chart(data.Opp)
#st.link_button("Google",url="google.com")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- App Title ---
st.title("Interactive Data Analysis & Classification")
