import streamlit as st

# Ustawienie layoutu na szeroki
st.set_page_config(layout="wide")

# Dodanie meta tag viewport
st.markdown(
    """
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    """,
    unsafe_allow_html=True
)

# Minimalny CSS
st.markdown(
    """
    <style>
    html, body, .block-container {
        height: 100%;
        width: 100%;
        margin: 0;
        padding: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Test Responsywności")

st.write("Jeśli widzisz ten tekst na całym ekranie, aplikacja jest responsywna.")
st.write("Spróbuj przewijać stronę, aby zobaczyć pełną funkcjonalność.")
