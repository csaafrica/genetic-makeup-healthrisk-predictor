import pandas as pd
import streamlit as st

bio_data = pd.read_csv(r'C:\Users\ryanj\Desktop\CSA2025\data\anthropometric_trait_gwas.csv')

st.write(bio_data.head())

