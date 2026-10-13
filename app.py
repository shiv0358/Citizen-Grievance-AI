import sqlite3
import pickle
import re
import unicodedata
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Citizen Grievance AI",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. DARK CHARCOAL AND CREAM THEME
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --background: #222226;
    --surface: #2B2B30;
    --surface-hover: #34343A;
    --primary: #FFD98E;
    --primary-hover: #F3C875;
    --text: #F2F0EA;
    --muted: #B5B3BC;
    --border: #414149;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--background);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.7rem;
    padding-bottom: 2.5rem;
}

h1, h2, h3, h4 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -0.02em;
}

h1 {
    font-size: 2.15rem !important;
    font-weight: 800 !important;
}

h2 {
    font-size: 1.5rem !important;
    font-weight: 750 !important;
}

h3 {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
}

p, label, span {
    color: inherit;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: #19191D;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 0.65rem 0.7rem;
    border-radius: 9px;
    margin: 0.12rem 0;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #34343A;
}

/* METRIC CARDS */

[data-testid="stMetric"] {
    background: var(--surface);
    padding: 1.05rem 1.15rem;
    border: 1px solid var(--border);
    border-radius: 13px;
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: var(--primary) !important;
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
}

/* BUTTONS */

.stButton > button,
.stDownloadButton > button,
[data-testid="stFormSubmitButton"] button {
    border-radius: 9px;
    font-weight: 650;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text);
    transition: 0.15s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: var(--primary);
    color: var(--primary);
}

.stButton > button[kind="primary"],
[data-testid="stFormSubmitButton"] button[kind="primary"] {
    background: var(--primary);
    color: #222226;
    border-color: var(--primary);
}

.stButton > button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
    background: var(--primary-hover);
    border-color: var(--primary-hover);
    color: #222226;
}

/* INPUT FIELDS */

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input {
    background: #29292E;
    color: var(--text);
    border-color: var(--border);
    border-radius: 8px;
}

[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder {
    color: #92909A;
}

/* TABLES */

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}

/* DIVIDERS */

hr {
    border-color: var(--border);
}

/* HOME HERO */

.hero {
    background: transparent;
    color: var(--text);
    padding: 0.4rem 0 1.1rem 0;
    margin-bottom: 1rem;
}

.hero h1 {
    color: var(--text) !important;
    margin: 0.5rem 0 0.7rem 0;
    font-size: 2.55rem !important;
}

.hero p {
    color: var(--muted) !important;
    max-width: 900px;
    font-size: 1rem;
    line-height: 1.7;
    margin-bottom: 0;
}

.hero-tag {
    color: var(--primary);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.74rem;
    font-weight: 700;
}

/* FEATURE CARDS */

.feature-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 1.15rem;
    min-height: 150px;
}

.feature-icon {
    font-size: 1.4rem;
    margin-bottom: 0.45rem;
}

.feature-title {
    font-family: 'Manrope', sans-serif;
    font-size: 1rem;
    font-weight: 800;
    color: var(--primary);
    margin-bottom: 0.35rem;
}

.feature-copy {
    color: var(--muted);
    font-size: 0.9rem;
    line-height: 1.55;
}

.muted-copy {
    color: var(--muted);
    font-size: 0.93rem;
}

/* EXPANDERS */

[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
}

/* ALERTS */

[data-testid="stAlert"] {
    border-radius: 10px;
}

/* FILE UPLOADER */

[data-testid="stFileUploader"] {
    background: var(--surface);
    border: 1px dashed var(--border);
    border-radius: 10px;
    padding: 0.6rem;
}

/* RESPONSIVE */

@media (max-width: 800px) {
    .block-container {
        padding-top: 1rem;
    }

    .hero h1 {
        font-size: 1.8rem !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. PROJECT PATHS AND CONSTANTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DB_NAME = BASE_DIR / "complaints.db"
MODEL_PATH = BASE_DIR / "complaint_model.pkl"
UPLOAD_DIR = BASE_DIR / "uploaded_images"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "Garbage Disposal",
    "Drainage / Sanitation",
    "Water Pollution",
    "Air Pollution",
    "Noise Pollution",
    "Other"
]

STATUSES = [
    "Pending",
    "In Progress",
    "Resolved"
]

PRIORITIES = [
    "High",
    "Medium",
    "Low"
]


# ============================================================
# 4. LOAD MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def load_model():
    """Load the trained complaint classification model."""

    if not MODEL_PATH.exists():
        return None

    try:
        with open(MODEL_PATH, "rb") as model_file:
            return pickle.load(model_file)
    except Exception:
        return None


model = load_model()


# ============================================================
# 5. DATABASE FUNCTIONS
# ============================================================

def initialize_database():
    """Create the database table if it does not exist."""

    with sqlite3.connect(DB_NAME) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                category TEXT,
                location TEXT,
                description TEXT,
                sentiment TEXT,
                priority TEXT,
                reason TEXT,
                status TEXT DEFAULT 'Pending',
                created_at TEXT
            )
        """)


def save_complaint(
    name,
    category,
    location,
    description,
    sentiment,
    priority,
    reason
):
    """Save a complaint and return its ID."""

    with sqlite3.connect(DB_NAME) as connection:
        cursor = connection.execute("""
            INSERT INTO complaints (
                name,
                category,
                location,
                description,
                sentiment,
                priority,
                reason,
                status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'Pending', ?)
        """, (
            name,
            category,
            location,
            description,
            sentiment,
            priority,
            reason,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        return cursor.lastrowid


def get_complaints():
    """Retrieve all complaints."""

    with sqlite3.connect(DB_NAME) as connection:
        return pd.read_sql_query(
            "SELECT * FROM complaints ORDER BY id DESC",
            connection
        )


def update_status(complaint_id, new_status):
    """Update the status of a complaint."""

    with sqlite3.connect(DB_NAME) as connection:
        connection.execute("""
            UPDATE complaints
            SET status = ?
            WHERE id = ?
        """, (new_status, complaint_id))


initialize_database()


# ============================================================
# 6. DEMONSTRATION COMPLAINTS
# ============================================================

# These are fictional examples for academic demonstrations.
# Existing database records are not deleted or overwritten.

DEMO_COMPLAINTS = [
    (
        "Demo Citizen 1",
        "Garbage Disposal",
        "Rajajinagar, Bengaluru",
        "[DEMO] Garbage has accumulated near the main road for several days.",
        "Negative / Concerned",
        "Medium",
        "Accumulated waste requires municipal review.",
        "Pending",
        "2026-10-11 09:15:00"
    ),
    (
        "Demo Citizen 2",
        "Drainage / Sanitation",
        "Vijayanagar, Bengaluru",
        "[DEMO] Sewage is overflowing from a blocked drain near residential buildings.",
        "Negative / Concerned",
        "High",
        "Potential sewage overflow requires prompt human review.",
        "In Progress",
        "2026-10-11 10:30:00"
    ),
    (
        "Demo Citizen 3",
        "Air Pollution",
        "Peenya, Bengaluru",
        "[DEMO] Thick smoke from an industrial area is affecting nearby residents.",
        "Negative / Concerned",
        "Medium",
        "Possible air pollution requires inspection.",
        "Pending",
        "2026-10-11 11:20:00"
    ),
    (
        "Demo Citizen 4",
        "Water Pollution",
        "Yelahanka, Bengaluru",
        "[DEMO] Water in the local lake appears discoloured and has an unusual smell.",
        "Negative / Concerned",
        "High",
        "Possible water contamination requires assessment.",
        "Pending",
        "2026-10-11 12:10:00"
    ),
    (
        "Demo Citizen 5",
        "Noise Pollution",
        "Indiranagar, Bengaluru",
        "[DEMO] Loud construction noise continues late at night near homes.",
        "Negative / Concerned",
        "Medium",
        "Noise complaint requires local authority review.",
        "Resolved",
        "2026-10-11 13:00:00"
    ),
    (
        "Demo Citizen 6",
        "Garbage Disposal",
        "Jayanagar, Bengaluru",
        "[DEMO] The municipal garbage collection vehicle has missed several scheduled visits.",
        "Negative / Concerned",
        "High",
        "Repeated missed waste collection requires review.",
        "Pending",
        "2026-10-11 14:00:00"
    ),
    (
        "Demo Citizen 7",
        "Drainage / Sanitation",
        "Whitefield, Bengaluru",
        "[DEMO] Rainwater is collecting on the road because the roadside drain is blocked.",
        "Negative / Concerned",
        "High",
        "Blocked drainage requires prompt inspection.",
        "Pending",
        "2026-10-11 14:30:00"
    ),
    (
        "Demo Citizen 8",
        "Water Pollution",
        "Hebbal, Bengaluru",
        "[DEMO] Floating waste is covering part of the neighbourhood lake.",
        "Negative / Concerned",
        "Medium",
        "Water pollution concern requires inspection.",
        "In Progress",
        "2026-10-11 15:00:00"
    ),
    (
        "Demo Citizen 9",
        "Air Pollution",
        "Electronic City, Bengaluru",
        "[DEMO] Construction dust is spreading into nearby residential buildings.",
        "Negative / Concerned",
        "Medium",
        "Dust pollution requires review.",
        "Pending",
        "2026-10-11 15:30:00"
    ),
    (
        "Demo Citizen 10",
        "Noise Pollution",
        "Marathahalli, Bengaluru",
        "[DEMO] Repeated loudspeaker noise is disturbing residents in the evening.",
        "Negative / Concerned",
        "Low",
        "Noise concern requires verification.",
        "Pending",
        "2026-10-11 16:00:00"
    ),
    (
        "Demo Citizen 11",
        "Garbage Disposal",
        "Malleshwaram, Bengaluru",
        "[DEMO] Household rubbish is scattered around the public dustbin.",
        "Negative / Concerned",
        "Low",
        "Waste collection and cleanup may be required.",
        "Resolved",
        "2026-10-11 16:30:00"
    ),
    (
        "Demo Citizen 12",
        "Drainage / Sanitation",
        "BTM Layout, Bengaluru",
        "[DEMO] A damaged drainage cover needs inspection near the footpath.",
        "Negative / Concerned",
        "Medium",
        "Damaged public infrastructure requires review.",
        "Pending",
        "2026-10-11 17:00:00"
    ),
    (
        "Demo Citizen 13",
        "Water Pollution",
        "Bellandur, Bengaluru",
        "[DEMO] An oily layer is visible on the surface of a local water body.",
        "Negative / Concerned",
        "High",
        "Possible contamination requires investigation.",
        "In Progress",
        "2026-10-11 17:30:00"
    ),
    (
        "Demo Citizen 14",
        "Air Pollution",
        "Banashankari, Bengaluru",
        "[DEMO] Smoke from open burning is affecting nearby houses.",
        "Negative / Concerned",
        "High",
        "Smoke exposure requires prompt assessment.",
        "Pending",
        "2026-10-11 18:00:00"
    ),
    (
        "Demo Citizen 15",
        "Noise Pollution",
        "Koramangala, Bengaluru",
        "[DEMO] Traffic horns are creating excessive noise near the residential area.",
        "Negative / Concerned",
        "Low",
        "Noise complaint requires review.",
        "Pending",
        "2026-10-11 18:30:00"
    )
]


def add_demo_complaints():
    """
    Insert each demo description only if it does not already exist.
    Existing complaints are preserved.
    """

    with sqlite3.connect(DB_NAME) as connection:

        for record in DEMO_COMPLAINTS:

            existing = connection.execute("""
                SELECT id
                FROM complaints
                WHERE description = ?
                LIMIT 1
            """, (record[3],)).fetchone()

            if existing:
                continue

            connection.execute("""
                INSERT INTO complaints (
                    name,
                    category,
                    location,
                    description,
                    sentiment,
                    priority,
                    reason,
                    status,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, record)


add_demo_complaints()


# ============================================================
# 7. CATEGORY PREDICTION
# ============================================================

def predict_category(description, selected_category):
    """Predict a category using the trained model when possible."""

    if model is None:
        return selected_category, False

    try:
        prediction = str(model.predict([description])[0])

        if prediction in CATEGORIES:
            return prediction, True

        return selected_category, False

    except Exception:
        return selected_category, False


# ============================================================
# 8. PRIORITY ANALYSIS
# ============================================================

def analyze_priority(description, location):
    """Estimate priority using simple predefined rules."""

    text = description.lower()
    place = location.lower()

    high_phrases = [
        "chemical leak",
        "toxic",
        "dangerous",
        "overflowing",
        "overflow",
        "drinking water",
        "sewage flooding",
        "gas leak",
        "exposed electric wire",
        "fire"
    ]

    medium_words = [
        "garbage",
        "waste",
        "smoke",
        "pollution",
        "leak",
        "dirty",
        "blocked",
        "rubbish",
        "noise"
    ]

    sensitive_locations = [
        "school",
        "hospital"
    ]

    if (
        any(term in text for term in high_phrases)
        or any(term in place for term in sensitive_locations)
    ):
        return (
            "High",
            "A potential hazard or sensitive location was identified. "
            "Human review is recommended."
        )

    if any(term in text for term in medium_words):
        return (
            "Medium",
            "The complaint describes a possible environmental or "
            "sanitation issue."
        )

    return (
        "Low",
        "The predefined rules did not identify a high-risk indicator. "
        "Human review is still required."
    )


# ============================================================
# 9. SENTIMENT ANALYSIS
# ============================================================

def analyze_sentiment(description):
    """Estimate complaint sentiment using predefined keywords."""

    negative_words = [
        "bad",
        "dirty",
        "terrible",
        "problem",
        "overflowing",
        "pollution",
        "smell",
        "dangerous",
        "unbearable",
        "unsafe",
        "disgusting",
        "harmful"
    ]

    text = description.lower()

    if any(word in text for word in negative_words):
        return "Negative / Concerned"

    return "Neutral or Unclear"


def analyze_complaint(description, selected_category, location):
    """Run category, sentiment, and priority analysis."""

    category, model_used = predict_category(
        description,
        selected_category
    )

    priority, reason = analyze_priority(
        description,
        location
    )

    sentiment = analyze_sentiment(description)

    return category, sentiment, priority, reason, model_used


# ============================================================
# 10. DUPLICATE COMPLAINT DETECTION
# ============================================================

def normalize_location(location):
    if not location:
        return ""

    location = unicodedata.normalize(
        "NFKC",
        str(location)
    ).lower().strip()

    location = re.sub(r"[^\w\s]", " ", location)

    return re.sub(r"\s+", " ", location)


def normalize_description(description):
    if not description:
        return ""

    description = unicodedata.normalize(
        "NFKC",
        str(description)
    ).lower()

    description = re.sub(r"[^\w\s]", " ", description)

    return re.sub(r"\s+", " ", description).strip()


def group_similar_complaints(
    complaints_df,
    similarity_threshold=0.45
):
    """
    Group complaints only when their normalized locations and
    categories match and their descriptions are sufficiently similar.
    """

    if complaints_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = complaints_df.copy().reset_index(drop=True)

    df["normalized_location"] = df["location"].apply(
        normalize_location
    )

    df["normalized_description"] = df["description"].apply(
        normalize_description
    )

    df = df[
        (df["normalized_location"] != "")
        & (df["normalized_description"] != "")
    ].copy().reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Compute text similarity.
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        matrix = cosine_similarity(
            vectorizer.fit_transform(
                df["normalized_description"].tolist()
            )
        )

    except ValueError:
        matrix = [
            [
                1.0 if i == j else 0.0
                for j in range(len(df))
            ]
            for i in range(len(df))
        ]

    # Union-find grouping.
    parent = list(range(len(df)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]

        return index

    def union(first, second):
        root_first = root(first)
        root_second = root(second)

        if root_first != root_second:
            parent[root_second] = root_first

    for i in range(len(df)):
        for j in range(i + 1, len(df)):

            same_location = (
                df.loc[i, "normalized_location"]
                == df.loc[j, "normalized_location"]
            )

            same_category = (
                df.loc[i, "category"]
                == df.loc[j, "category"]
            )

            similar_text = (
                matrix[i][j] >= similarity_threshold
            )

            if same_location and same_category and similar_text:
                union(i, j)

    mapping = {}
    groups = []

    for i in range(len(df)):
        group_root = root(i)

        if group_root not in mapping:
            mapping[group_root] = len(mapping) + 1

        groups.append(mapping[group_root])

    df["issue_group"] = groups

    priority_rank = {
        "Low": 1,
        "Medium": 2,
        "High": 3
    }

    rows = []

    for group_id, group in df.groupby("issue_group"):

        count = len(group)

        highest_priority = max(
            group["priority"].tolist(),
            key=lambda value: priority_rank.get(value, 1)
        )

        if count >= 5:
            suggested_priority = "High"

        elif count >= 3 and highest_priority == "High":
            suggested_priority = "High"

        elif count >= 3:
            suggested_priority = "Medium"

        else:
            suggested_priority = highest_priority

        rows.append({
            "Issue Group": f"Issue #{group_id}",
            "Location": group["location"].iloc[0],
            "Category": group["category"].iloc[0],
            "Number of Reports": count,
            "Suggested Priority": suggested_priority,
            "Complaint IDs": ", ".join(
                map(str, group["id"].tolist())
            )
        })

    summary = pd.DataFrame(rows)

    if not summary.empty:
        summary = summary.sort_values(
            "Number of Reports",
            ascending=False
        ).reset_index(drop=True)

    return summary, df


def get_duplicate_analysis(complaints_df):
    """
    Return duplicate groups, records in duplicate groups,
    number of complaints in repeated groups, number of groups,
    and number of non-duplicated complaints.
    """

    if complaints_df.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            0,
            0,
            0
        )

    summary, grouped = group_similar_complaints(
        complaints_df,
        similarity_threshold=0.45
    )

    if summary.empty or grouped.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            0,
            0,
            len(complaints_df)
        )

    duplicate_groups = summary[
        summary["Number of Reports"] > 1
    ].copy()

    if duplicate_groups.empty:
        return (
            duplicate_groups,
            pd.DataFrame(),
            0,
            0,
            len(complaints_df)
        )

    group_ids = {
        int(str(value).replace("Issue #", ""))
        for value in duplicate_groups["Issue Group"]
    }

    duplicate_records = grouped[
        grouped["issue_group"].isin(group_ids)
    ].copy()

    # Count every complaint in a repeated group.
    repeated_count = len(duplicate_records)

    # Complaints outside repeated groups.
    non_duplicate_count = (
        len(complaints_df) - repeated_count
    )

    return (
        duplicate_groups,
        duplicate_records,
        repeated_count,
        len(duplicate_groups),
        non_duplicate_count
    )


# ============================================================
# 11. SHARED PAGE ELEMENTS
# ============================================================

def page_header(title, subtitle=""):
    st.title(title)

    if subtitle:
        st.markdown(
            f'<div class="muted-copy">{subtitle}</div>',
            unsafe_allow_html=True
        )


def sort_by_priority(dataframe):
    """Sort complaints High, Medium, Low, then newest first."""

    priority_order = {
        "High": 0,
        "Medium": 1,
        "Low": 2
    }

    result = dataframe.copy()

    result["_priority_order"] = (
        result["priority"]
        .map(priority_order)
        .fillna(3)
    )

    result = result.sort_values(
        by=["_priority_order", "created_at", "id"],
        ascending=[True, False, False]
    )

    return result.drop(columns=["_priority_order"])


# ============================================================
# 12. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown("## Citizen Grievance AI")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Submit Complaint",
            "Admin Dashboard",
            "Duplicate Complaints",
            "About & Contact Authorities"
        ],
        label_visibility="collapsed",
        key="main_navigation"
    )


# ============================================================
# 13. HOME PAGE
# ============================================================

if page == "Home":

    df = get_complaints()

    total = len(df)

    high = (
        int((df["priority"] == "High").sum())
        if total else 0
    )

    resolved = (
        int((df["status"] == "Resolved").sum())
        if total else 0
    )

    pending = (
        int((df["status"] == "Pending").sum())
        if total else 0
    )

    st.markdown("""
    <div class="hero">
        <h1>Make every complaint count.</h1>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Complaints", f"{total:,}")
    col2.metric("High Priority", f"{high:,}")
    col3.metric("Resolved", f"{resolved:,}")
    col4.metric("Pending Review", f"{pending:,}")

    st.markdown("## Platform Features")

    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">AI-Assisted Analysis</div>
            <div class="feature-copy">
                Predict complaint categories using a trained model
                when available and estimate priority using rules.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Duplicate Detection</div>
            <div class="feature-copy">
                Compare locations, categories, and text similarity
                to identify potentially repeated reports.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Interactive Analytics</div>
            <div class="feature-copy">
                Explore complaint priorities, categories, locations,
                and resolution status through charts.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## Recent Complaints")

    if df.empty:

        st.info("No complaints have been submitted yet.")

    else:

        recent = df.head(5)[
            [
                "id",
                "category",
                "location",
                "priority",
                "status",
                "created_at"
            ]
        ].rename(columns={
            "id": "Complaint ID",
            "category": "Category",
            "location": "Location",
            "priority": "Priority",
            "status": "Status",
            "created_at": "Submitted At"
        })

        st.dataframe(
            recent,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# 14. SUBMIT COMPLAINT PAGE
# ============================================================

elif page == "Submit Complaint":

    page_header("Submit a Complaint")

    with st.form(
        "submit_complaint_form",
        clear_on_submit=True
    ):

        name = st.text_input(
            "Full Name",
            placeholder="Enter your name"
        )

        category = st.selectbox(
            "Complaint Category",
            CATEGORIES
        )

        location = st.text_input(
            "Location",
            placeholder="Example: Rajajinagar, Bengaluru"
        )

        description = st.text_area(
            "Describe the Complaint",
            placeholder=(
                "Explain what happened, where it happened, "
                "and how it affects the public..."
            ),
            height=150,
            max_chars=3000
        )

        uploaded_image = st.file_uploader(
            "Supporting Image (Optional)",
            type=["png", "jpg", "jpeg"]
        )

        submitted = st.form_submit_button(
            "Submit Complaint",
            type="primary",
            use_container_width=True
        )

    if submitted:

        if not name.strip():

            st.error("Please enter your name.")

        elif not location.strip():

            st.error("Please enter the complaint location.")

        elif not description.strip():

            st.error("Please describe the complaint.")

        else:

            (
                predicted_category,
                sentiment,
                priority,
                reason,
                model_used
            ) = analyze_complaint(
                description,
                category,
                location
            )

            complaint_id = save_complaint(
                name.strip(),
                predicted_category,
                location.strip(),
                description.strip(),
                sentiment,
                priority,
                reason
            )

            saved_image_path = None

            if uploaded_image is not None:

                safe_name = re.sub(
                    r"[^A-Za-z0-9_.-]",
                    "_",
                    uploaded_image.name
                )

                filename = (
                    f"complaint_{complaint_id}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_"
                    f"{safe_name}"
                )

                saved_image_path = UPLOAD_DIR / filename

                saved_image_path.write_bytes(
                    uploaded_image.getbuffer()
                )

            st.success(
                f"Complaint submitted successfully. "
                f"Complaint ID: #{complaint_id}"
            )

            r1, r2, r3 = st.columns(3)

            r1.metric("Category", predicted_category)
            r2.metric("Estimated Priority", priority)
            r3.metric("Sentiment", sentiment)

            st.write(f"**Priority Explanation:** {reason}")
            st.write("**Current Status:** Pending")

            if saved_image_path:

                st.image(
                    str(saved_image_path),
                    caption="Uploaded supporting image",
                    use_container_width=True
                )


# ============================================================
# 15. ADMIN DASHBOARD
# ============================================================

elif page == "Admin Dashboard":

    page_header("Admin Dashboard")

    df = get_complaints()

    if df.empty:

        st.info("No complaints have been submitted yet.")

    else:

        # ----------------------------------------------------
        # DASHBOARD METRICS
        # ----------------------------------------------------

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Total", len(df))

        c2.metric(
            "High Priority",
            int((df["priority"] == "High").sum())
        )

        c3.metric(
            "Pending",
            int((df["status"] == "Pending").sum())
        )

        c4.metric(
            "In Progress",
            int((df["status"] == "In Progress").sum())
        )

        c5.metric(
            "Resolved",
            int((df["status"] == "Resolved").sum())
        )

        # ----------------------------------------------------
        # SEARCH AND FILTER
        # ----------------------------------------------------

        st.markdown("## Search and Filter")

        f1, f2, f3 = st.columns(3)

        with f1:

            query = st.text_input(
                "Search",
                placeholder="Name, location, description..."
            )

        with f2:

            category_filter = st.selectbox(
                "Category",
                ["All"] + CATEGORIES
            )

        with f3:

            status_filter = st.selectbox(
                "Status",
                ["All"] + STATUSES
            )

        filtered = df.copy()

        if query.strip():

            q = query.strip().lower()

            searchable_columns = [
                "name",
                "location",
                "description",
                "category"
            ]

            mask = pd.Series(
                False,
                index=filtered.index
            )

            for col in searchable_columns:

                mask |= (
                    filtered[col]
                    .fillna("")
                    .astype(str)
                    .str.lower()
                    .str.contains(q, regex=False)
                )

            filtered = filtered[mask]

        if category_filter != "All":

            filtered = filtered[
                filtered["category"] == category_filter
            ]

        if status_filter != "All":

            filtered = filtered[
                filtered["status"] == status_filter
            ]

        # ----------------------------------------------------
        # TABLE 1: COMPLAINT RECORDS
        # ----------------------------------------------------

        st.markdown("## Complaint Records")

        table_columns = [
            "id",
            "name",
            "category",
            "location",
            "priority",
            "sentiment",
            "status",
            "created_at"
        ]

        display = filtered[table_columns].rename(columns={
            "id": "Complaint ID",
            "name": "Citizen Name",
            "category": "Category",
            "location": "Location",
            "priority": "Priority",
            "sentiment": "Sentiment",
            "status": "Status",
            "created_at": "Submitted At"
        })

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Filtered Complaints (CSV)",
            filtered.to_csv(index=False).encode("utf-8"),
            "citizen_complaints.csv",
            "text/csv",
            key="download_filtered_complaints"
        )

        # ----------------------------------------------------
        # TABLE 2: PRIORITY-WISE COMPLAINTS
        # ----------------------------------------------------

        st.divider()

        st.markdown("## Priority-Wise Complaints")

        priority_sorted = sort_by_priority(filtered)

        priority_display = priority_sorted[
            [
                "id",
                "name",
                "category",
                "location",
                "description",
                "priority",
                "status",
                "created_at"
            ]
        ].rename(columns={
            "id": "Complaint ID",
            "name": "Citizen Name",
            "category": "Category",
            "location": "Location",
            "description": "Description",
            "priority": "Priority",
            "status": "Status",
            "created_at": "Submitted At"
        })

        st.dataframe(
            priority_display,
            use_container_width=True,
            hide_index=True,
            height=420
        )

        st.download_button(
            "Download Priority-Sorted Complaints (CSV)",
            priority_display.to_csv(
                index=False
            ).encode("utf-8"),
            "priority_sorted_complaints.csv",
            "text/csv",
            key="download_priority_sorted"
        )

        # ----------------------------------------------------
        # UPDATE COMPLAINT STATUS
        # ----------------------------------------------------

        st.divider()

        st.markdown("## Update Complaint Status")

        if filtered.empty:

            st.caption(
                "No complaints match the current filters."
            )

        else:

            with st.form("update_status_form"):

                u1, u2 = st.columns(2)

                with u1:

                    selected_id = st.selectbox(
                        "Complaint ID",
                        filtered["id"].tolist()
                    )

                current_status = df.loc[
                    df["id"] == selected_id,
                    "status"
                ].iloc[0]

                with u2:

                    new_status = st.selectbox(
                        "New Status",
                        STATUSES,
                        index=(
                            STATUSES.index(current_status)
                            if current_status in STATUSES
                            else 0
                        )
                    )

                update_clicked = st.form_submit_button(
                    "Save Status",
                    type="primary"
                )

            if update_clicked:

                update_status(
                    selected_id,
                    new_status
                )

                st.success(
                    f"Complaint #{selected_id} updated to {new_status}."
                )

                st.rerun()

        # ----------------------------------------------------
        # ANALYTICS CHARTS
        # ----------------------------------------------------

        st.markdown("## Complaint Analytics")

        chart1, chart2 = st.columns(2)

        with chart1:

            st.markdown("### Priority Distribution")

            counts = (
                df["priority"]
                .value_counts()
                .reindex(PRIORITIES, fill_value=0)
                .rename_axis("Priority")
                .reset_index(name="Complaints")
            )

            fig = px.pie(
                counts,
                names="Priority",
                values="Complaints",
                hole=0.58,
                color="Priority",
                color_discrete_map={
                    "High": "#E87979",
                    "Medium": "#E6B566",
                    "Low": "#FFD98E"
                }
            )

            fig.update_traces(
                textposition="inside",
                textinfo="percent+label"
            )

            fig.update_layout(
                height=350,
                margin=dict(l=8, r=8, t=20, b=8),
                paper_bgcolor="#2B2B30",
                plot_bgcolor="#2B2B30",
                font=dict(color="#F2F0EA"),
                legend=dict(font=dict(color="#F2F0EA"))
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with chart2:

            st.markdown("### Complaints by Category")

            counts = (
                df["category"]
                .value_counts()
                .sort_values()
                .rename_axis("Category")
                .reset_index(name="Complaints")
            )

            fig = px.bar(
                counts,
                x="Complaints",
                y="Category",
                orientation="h",
                text="Complaints",
                color_discrete_sequence=["#FFD98E"]
            )

            fig.update_traces(
                textposition="outside",
                cliponaxis=False
            )

            fig.update_layout(
                height=350,
                margin=dict(l=8, r=25, t=20, b=8),
                paper_bgcolor="#2B2B30",
                plot_bgcolor="#2B2B30",
                xaxis_title="Number of Complaints",
                yaxis_title="",
                font=dict(color="#F2F0EA"),
                xaxis=dict(
                    gridcolor="#414149",
                    zerolinecolor="#414149"
                ),
                yaxis=dict(
                    gridcolor="#414149",
                    zerolinecolor="#414149"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ----------------------------------------------------
        # MOST REPORTED LOCATIONS
        # ----------------------------------------------------

        st.markdown("### Most Reported Locations")

        location_counts = (
            df["location"]
            .fillna("Unknown")
            .replace("", "Unknown")
            .value_counts()
            .head(10)
            .sort_values()
            .rename_axis("Location")
            .reset_index(name="Complaints")
        )

        fig = px.bar(
            location_counts,
            x="Complaints",
            y="Location",
            orientation="h",
            text="Complaints",
            color_discrete_sequence=["#D6C6A4"]
        )

        fig.update_traces(
            textposition="outside",
            cliponaxis=False
        )

        fig.update_layout(
            height=390,
            margin=dict(l=8, r=25, t=20, b=8),
            paper_bgcolor="#2B2B30",
            plot_bgcolor="#2B2B30",
            xaxis_title="Number of Complaints",
            yaxis_title="",
            font=dict(color="#F2F0EA"),
            xaxis=dict(
                gridcolor="#414149",
                zerolinecolor="#414149"
            ),
            yaxis=dict(
                gridcolor="#414149",
                zerolinecolor="#414149"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ----------------------------------------------------
        # RESOLUTION STATUS
        # ----------------------------------------------------

        st.markdown("### Resolution Status")

        counts = (
            df["status"]
            .value_counts()
            .reindex(STATUSES, fill_value=0)
            .rename_axis("Status")
            .reset_index(name="Complaints")
        )

        fig = px.bar(
            counts,
            x="Status",
            y="Complaints",
            text="Complaints",
            color="Status",
            color_discrete_map={
                "Pending": "#E6B566",
                "In Progress": "#C6B6E8",
                "Resolved": "#A8C9A2"
            }
        )

        fig.update_traces(
            textposition="outside",
            cliponaxis=False
        )

        fig.update_layout(
            height=320,
            margin=dict(l=8, r=15, t=20, b=8),
            showlegend=False,
            paper_bgcolor="#2B2B30",
            plot_bgcolor="#2B2B30",
            yaxis_title="Number of Complaints",
            xaxis_title="",
            font=dict(color="#F2F0EA"),
            yaxis=dict(
                gridcolor="#414149",
                zerolinecolor="#414149"
            ),
            xaxis=dict(
                gridcolor="#414149",
                zerolinecolor="#414149"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# 16. DUPLICATE COMPLAINTS PAGE
# ============================================================

elif page == "Duplicate Complaints":

    page_header("Complaint Summary")

    df = get_complaints()

    if df.empty:

        st.info("No complaints have been submitted yet.")

    else:

        (
            duplicate_groups,
            duplicate_records,
            repeated_count,
            group_count,
            non_duplicate_count
        ) = get_duplicate_analysis(df)

        # ----------------------------------------------------
        # THREE SIMPLE SUMMARY CARDS
        # ----------------------------------------------------

        st.markdown("## Complaint Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Complaints",
            len(df)
        )

        col2.metric(
            "Duplicated or Repeated Issues",
            repeated_count
        )

        col3.metric(
            "Non-Duplicated Complaints",
            non_duplicate_count
        )

        st.divider()

        # ----------------------------------------------------
        # REPEATED ISSUES
        # ----------------------------------------------------

        st.markdown("## Duplicated or Repeated Issues")

        if duplicate_groups.empty:

            st.info(
                "No repeated-issue groups were identified. "
                "All complaints are currently treated as non-duplicated."
            )

        else:

            repeated_display = duplicate_groups[
                [
                    "Issue Group",
                    "Location",
                    "Category",
                    "Number of Reports",
                    "Suggested Priority",
                    "Complaint IDs"
                ]
            ].rename(columns={
                "Number of Reports": "Number of Complaints"
            })

            st.dataframe(
                repeated_display,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### Complaints in Repeated Groups")

            repeated_records_display = duplicate_records[
                [
                    "id",
                    "name",
                    "category",
                    "location",
                    "description",
                    "priority",
                    "status",
                    "created_at",
                    "issue_group"
                ]
            ].rename(columns={
                "id": "Complaint ID",
                "name": "Citizen Name",
                "category": "Category",
                "location": "Location",
                "description": "Description",
                "priority": "Priority",
                "status": "Status",
                "created_at": "Submitted At",
                "issue_group": "Issue Group"
            })

            st.dataframe(
                repeated_records_display,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Repeated Complaints (CSV)",
                repeated_records_display.to_csv(
                    index=False
                ).encode("utf-8"),
                "repeated_complaints.csv",
                "text/csv",
                key="download_repeated_complaints"
            )

        st.divider()

        # ----------------------------------------------------
        # NON-DUPLICATED COMPLAINTS
        # ----------------------------------------------------

        st.markdown("## Non-Duplicated Complaints")

        if duplicate_records.empty:

            non_duplicate_records = df.copy()

        else:

            repeated_ids = set(
                duplicate_records["id"].tolist()
            )

            non_duplicate_records = df[
                ~df["id"].isin(repeated_ids)
            ].copy()

        if non_duplicate_records.empty:

            st.info(
                "Every complaint currently belongs to a "
                "potential repeated-issue group."
            )

        else:

            non_duplicate_display = non_duplicate_records[
                [
                    "id",
                    "name",
                    "category",
                    "location",
                    "description",
                    "priority",
                    "status",
                    "created_at"
                ]
            ].rename(columns={
                "id": "Complaint ID",
                "name": "Citizen Name",
                "category": "Category",
                "location": "Location",
                "description": "Description",
                "priority": "Priority",
                "status": "Status",
                "created_at": "Submitted At"
            })

            st.dataframe(
                non_duplicate_display,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Non-Duplicated Complaints (CSV)",
                non_duplicate_display.to_csv(
                    index=False
                ).encode("utf-8"),
                "non_duplicated_complaints.csv",
                "text/csv",
                key="download_non_duplicated_complaints"
            )


# ============================================================
# 17. ABOUT AND CONTACT AUTHORITIES
# ============================================================

elif page == "About & Contact Authorities":

    page_header("About & Contact Authorities")

    st.markdown("## Project Overview")

    left, right = st.columns(2)

    with left:

        st.markdown("### Project Objectives")

        st.markdown("""
        - Register public complaints.
        - Organize reports by category and location.
        - Estimate urgency for human review.
        - Identify potentially repeated reports.
        - Track resolution status.
        - Visualize complaint trends.
        """)

    with right:

        st.markdown("### Technologies Used")

        st.markdown("""
        - **Python** — application logic
        - **Streamlit** — web interface
        - **SQLite** — local database
        - **Pandas** — data processing
        - **Scikit-learn** — classification and text similarity
        - **Plotly** — interactive charts
        """)

    st.divider()

    st.markdown("## Important Contact Numbers")

    st.warning(
        "Use emergency services only for genuine emergencies. "
        "The numbers below serve different purposes."
    )

    contact1, contact2 = st.columns(2)

    with contact1:

        st.markdown("### Emergency Services")

        st.markdown("**112 — Emergency Response Support System**")

        st.write(
            "For urgent situations requiring police, fire, or other "
            "emergency assistance."
        )

        st.markdown("**National Consumer Helpline: 1915**")

        st.write(
            "For consumer-related complaints involving products and "
            "services. This is not a general municipal garbage or "
            "drainage helpline."
        )

    with contact2:

        st.markdown("### Civic and Municipal Complaints")

        st.write(
            "For garbage collection, blocked drains, sanitation, "
            "street-level public issues, and local environmental concerns, "
            "contact your municipal corporation or the relevant local department."
        )

        st.write(
            "Municipal complaint numbers can vary by city and department. "
            "Confirm the correct local contact before calling."
        )

    st.markdown("### Useful Official Resources")

    st.markdown(
        "- **Emergency Response Support System:** "
        "[112 India](https://112.gov.in/)"
    )

    st.markdown(
        "- **National Consumer Helpline:** "
        "[consumerhelpline.gov.in](https://consumerhelpline.gov.in/)"
    )

    st.markdown(
        "- **Swachhata-related civic reporting:** "
        "[Swachhata platform](https://dashboard.swachh.city/)"
    )

    st.divider()

    st.markdown("## Which Authority Should You Contact?")

    with st.expander("Garbage Disposal and Waste"):

        st.write(
            "Contact your municipal corporation or local waste-management "
            "department for missed garbage collection, overflowing bins, "
            "and waste accumulation."
        )

    with st.expander("Drainage and Sanitation"):

        st.write(
            "Contact the municipal sanitation, sewerage, or water-board "
            "department for blocked drains, sewage overflow, and sanitation issues."
        )

    with st.expander("Water Pollution"):

        st.write(
            "For suspected water contamination, contact the relevant water "
            "supply authority or pollution-control authority. Avoid drinking "
            "water that you reasonably suspect is contaminated."
        )

    with st.expander("Air Pollution"):

        st.write(
            "For smoke, dust, or suspected industrial emissions, contact "
            "the relevant municipal or pollution-control authority."
        )

    with st.expander("Noise Pollution"):

        st.write(
            "For excessive noise, contact the relevant local authority. "
            "Where immediate intervention is required, contact the appropriate "
            "police or emergency service."
        )