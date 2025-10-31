import streamlit as st
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Streamlit setup ---
st.set_page_config(page_title="Object Code Finder", layout="wide")

# --- Paths ---
file_path = Path(r"C:\Users\jtherman\OneDrive - University of Arizona\Projects\Object_Codes\Object_Codes.xlsx")
logo_path = Path(r"C:\Users\jtherman\OneDrive - University of Arizona\Projects\Object_Codes\FSSLogo.png")

# --- Header with logo ---
col1, col2 = st.columns([1, 10])
with col1:
    if logo_path.exists():
        st.image(str(logo_path), width=120)
with col2:
    st.markdown("<h1 style='padding-top: 15px;'>🔍 Object Code Finder</h1>", unsafe_allow_html=True)

# --- Load Excel file ---
if not file_path.exists():
    st.error(f"❌ Excel file not found at: {file_path}")
    st.stop()

df = pd.read_excel(file_path, sheet_name="Sheet1")

# --- Format Object Codes to always be 4 digits ---
if "Object Code" in df.columns:
    df["Object Code"] = df["Object Code"].astype(str).str.zfill(4)


# --- Cache model load ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()

# --- Combine searchable text ---
df["search_text"] = df["Name"].fillna('') + " " + df["Description"].fillna('')
corpus_embeddings = model.encode(df["search_text"].tolist(), show_progress_bar=False)


# --- Helper: Convert similarity score to gold star SVGs ---
def stars_svg(score: float) -> str:
    """Return HTML with gold SVG stars (supports half stars, no newlines)."""
    full_star_svg = '<svg width="20" height="20" viewBox="0 0 24 24" fill="#FFD700" xmlns="http://www.w3.org/2000/svg"><path d="M12 .587l3.668 7.571L24 9.748l-6 5.848L19.335 24 12 20.012 4.665 24 6 15.596 0 9.748l8.332-1.59z"/></svg>'
    half_star_svg = (
        '<svg width="20" height="20" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">'
        '<defs><linearGradient id="half">'
        '<stop offset="50%" stop-color="#FFD700" />'
        '<stop offset="50%" stop-color="#ddd" />'
        '</linearGradient></defs>'
        '<path fill="url(#half)" d="M12 .587l3.668 7.571L24 9.748l-6 5.848L19.335 24 '
        '12 20.012 4.665 24 6 15.596 0 9.748l8.332-1.59z"/>'
        '</svg>'
    )
    empty_star_svg = '<svg width="20" height="20" viewBox="0 0 24 24" fill="#ddd" xmlns="http://www.w3.org/2000/svg"><path d="M12 .587l3.668 7.571L24 9.748l-6 5.848L19.335 24 12 20.012 4.665 24 6 15.596 0 9.748l8.332-1.59z"/></svg>'

    stars = int(score)
    half_star = (score - stars) >= 0.5
    return (
            full_star_svg * stars
            + (half_star_svg if half_star else "")
            + empty_star_svg * (5 - stars - (1 if half_star else 0))
    )


# --- Search section ---
# Place the type selectbox UNDERNEATH the search box (one column layout)
user_input = st.text_input(
    "Enter a keyword, description, or related term (e.g. 'UPS', 'meals', 'office supplies'):"
)
type_options = ["All"] + sorted(df["Type"].dropna().unique().tolist())
selected_type = st.selectbox("Filter by Type:", type_options, label_visibility="visible")

# --- Hybrid search (Literal + Semantic) ---
if user_input:
    query = user_input.strip().lower()

    # 1️⃣ Literal matches (exact text search)
    literal_matches = df[
        df["Object Code"].str.contains(query, case=False, na=False)
        | df["Name"].str.contains(query, case=False, na=False)
        | df["Description"].str.contains(query, case=False, na=False)
        ].copy()
    literal_matches["similarity"] = 1.0
    literal_matches["Search Accuracy"] = stars_svg(5)

    # 2️⃣ Semantic matches (AI similarity)
    query_embedding = model.encode([user_input])
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
    df["similarity"] = similarities


    def score_to_star_value(score):
        if score >= 0.7:
            return 5.0
        elif score >= 0.61665:
            return 4.5
        elif score >= 0.5333:
            return 4.0
        elif score >= 0.45:
            return 3.5
        elif score >= 0.3667:
            return 3.0
        elif score >= 0.28335:
            return 2.5
        elif score >= 0.2:
            return 2.0
        elif score >= 0.11665:
            return 1.5
        else:
            return 1.0


    semantic_matches = df.copy()
    semantic_matches["Search Accuracy"] = semantic_matches["similarity"].apply(lambda s: stars_svg(score_to_star_value(s)))
    semantic_matches = semantic_matches.sort_values("similarity", ascending=False).head(25)

    # Merge results (literal first, then semantic)
    combined = pd.concat([literal_matches, semantic_matches])
    combined = combined.drop_duplicates(subset=["Object Code"]).reset_index(drop=True)

    # Apply Type filter
    if selected_type != "All":
        combined = combined[combined["Type"] == selected_type]

    # --- Display results ---
    if not combined.empty:
        st.markdown(f"**Showing {len(combined)} results (Literal matches first, then semantic suggestions):**")

        display_df = combined[["Object Code", "Name", "Description", "Type", "Search Accuracy"]]

        styled_html = display_df.to_html(
            index=False, escape=False, justify="left", classes="wrapped-table"
        )

        st.markdown(
            """
            <style>
            .wrapped-table {
                border-collapse: collapse;
                width: 100%;
            }
            .wrapped-table th, .wrapped-table td {
                text-align: left;
                border: 1px solid #ddd;
                padding: 6px;
                vertical-align: top;
                white-space: normal !important;
                word-wrap: break-word !important;
                max-width: 480px;
            }
            .wrapped-table th {
                background-color: #f4f4f4;
                font-weight: bold;
            }
            .stSelectbox [data-baseweb="select"] {
                max-width: 180px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(styled_html, unsafe_allow_html=True)
    else:
        st.warning("No matches found.")
