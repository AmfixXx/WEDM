import streamlit as st

# 1. Ustawienie layoutu na 'wide'
st.set_page_config(layout="wide")

# 2. Dodanie meta tag viewport i szczegółowego CSS dla pełnego wykorzystania ekranu
st.markdown(
    """
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
    /* Upewnij się, że html, body i główne kontenery zajmują 100% wysokości i szerokości */
    html, body, .stApp, .block-container, .main {
        height: 100%;
        width: 100%;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    /* Użyj Flexboxa do rozciągnięcia zawartości */
    .stApp {
        display: flex;
        flex-direction: column;
    }

    .block-container {
        flex: 1;
        display: flex;
        flex-direction: column;
        overflow: auto;
    }

    /* Usuń dodatkowe paddingi wewnątrz kontenerów */
    .block-container > div {
        padding: 0 !important;
    }

    /* Ustaw tło na biały kolor */
    body {
        background-color: white;
    }

    /* Upewnij się, że tytuł i tekst mają elastyczną wysokość */
    .element-container {
        flex: 1;
    }

    /* Usuń marginesy i paddingi dla tytułów i tekstów */
    .stTitle, .stText {
        margin: 0;
        padding: 0;
    }

    /* Dodatkowe ustawienia, aby zapewnić pełną responsywność */
    .stButton > button {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 3. Minimalny interfejs aplikacji Streamlit
st.title("Test Responsywności")

st.write("Jeśli widzisz ten tekst na całym ekranie, aplikacja jest responsywna.")
st.write("Spróbuj przewijać stronę, aby zobaczyć pełną funkcjonalność.")
