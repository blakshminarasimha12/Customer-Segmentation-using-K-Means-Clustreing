import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


st.set_page_config(
    page_title="Customer Segmentation using K-Means",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Segmentation using K-Means Clustering")
st.write(
    "Segment customers based on age, annual income, spending score, "
    "and purchase frequency using unsupervised machine learning."
)



@st.cache_data
def create_demo_dataset(n=300, seed=42):
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 61, n)

    gender = rng.choice(
        ["Male", "Female"],
        size=n
    )

    income = np.clip(
        rng.normal(60000, 22000, n),
        15000,
        150000
    ).round().astype(int)

    spending = np.clip(
        48
        + 0.00025 * (income - 60000)
        - 0.35 * (age - 35)
        + rng.normal(0, 15, n),
        1,
        100
    ).round().astype(int)

    frequency = np.clip(
        3
        + spending / 12
        + rng.normal(0, 2.5, n),
        1,
        30
    ).round().astype(int)

    return pd.DataFrame({
        "Customer_ID": np.arange(1001, 1001 + n),
        "Age": age,
        "Gender": gender,
        "Annual_Income": income,
        "Spending_Score": spending,
        "Purchase_Frequency": frequency
    })


st.sidebar.header("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload Customer CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    data_source = "Uploaded CSV"
else:
    df = create_demo_dataset()
    data_source = "Built-in demonstration dataset"

st.sidebar.info(
    f"Data source: {data_source}"
)


required_columns = [
    "Customer_ID",
    "Age",
    "Gender",
    "Annual_Income",
    "Spending_Score",
    "Purchase_Frequency"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "The uploaded CSV is missing these columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

df = df.copy()

df = df.drop_duplicates()

numeric_columns = [
    "Age",
    "Annual_Income",
    "Spending_Score",
    "Purchase_Frequency"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df["Gender"] = (
    df["Gender"]
    .astype(str)
    .str.strip()
    .str.title()
)


for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )


df.loc[df["Age"] < 0, "Age"] = np.nan
df.loc[df["Annual_Income"] < 0, "Annual_Income"] = np.nan

df["Spending_Score"] = df[
    "Spending_Score"
].clip(1, 100)

df["Purchase_Frequency"] = df[
    "Purchase_Frequency"
].clip(lower=0)

df = df.dropna(
    subset=numeric_columns
).reset_index(drop=True)


features = [
    "Age",
    "Annual_Income",
    "Spending_Score",
    "Purchase_Frequency"
]

X = df[features]


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)



k = st.sidebar.slider(
    "Number of Clusters (K)",
    min_value=2,
    max_value=8,
    value=4,
    step=1
)


model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

labels = model.fit_predict(X_scaled)

df["Cluster"] = labels + 1


summary = (
    df
    .groupby("Cluster")
    .agg(
        Age=("Age", "mean"),
        Annual_Income=("Annual_Income", "mean"),
        Spending_Score=("Spending_Score", "mean"),
        Purchase_Frequency=("Purchase_Frequency", "mean"),
        Customer_Count=("Customer_ID", "count")
    )
    .round(2)
)

summary["Percentage"] = (
    summary["Customer_Count"]
    / len(df)
    * 100
).round(2)



income_median = summary["Annual_Income"].median()
spending_median = summary["Spending_Score"].median()
frequency_median = summary["Purchase_Frequency"].median()


def get_segment_name(row):
    high_income = row["Annual_Income"] >= income_median
    high_spending = row["Spending_Score"] >= spending_median
    high_frequency = row["Purchase_Frequency"] >= frequency_median

    if high_income and high_spending and high_frequency:
        return "High-Value Loyal Customers"

    if high_income and high_spending:
        return "Premium Customers"

    if high_income and not high_spending:
        return "Affluent Low-Engagement Customers"

    if not high_income and high_spending and high_frequency:
        return "Frequent Budget Customers"

    if not high_income and high_spending:
        return "Potential Growth Customers"

    if not high_income and not high_spending:
        return "Budget Customers"

    return "Moderate Customers"


segment_names = {}

for cluster, row in summary.iterrows():
    segment_names[cluster] = get_segment_name(row)

df["Segment_Name"] = df[
    "Cluster"
].map(segment_names)

summary_display = summary.copy()

summary_display.insert(
    0,
    "Segment",
    [
        segment_names[c]
        for c in summary.index
    ]
)


if len(df) > k:
    silhouette = silhouette_score(
        X_scaled,
        labels
    )
else:
    silhouette = 0


st.subheader("📈 Project Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    len(df)
)

col2.metric(
    "Number of Clusters",
    k
)

col3.metric(
    "Average Income",
    f"₹{df['Annual_Income'].mean():,.0f}"
)

col4.metric(
    "Silhouette Score",
    f"{silhouette:.3f}"
)

st.divider()



tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Cluster Analysis",
    "📈 Visualizations",
    "📉 Elbow Method",
    "👥 Customer Data",
    "💡 Business Insights"
])


with tab1:

    st.subheader("Customer Cluster Summary")

    st.dataframe(
        summary_display,
        use_container_width=True
    )

    st.subheader("Cluster Characteristics")

    for cluster in summary.index:

        row = summary.loc[cluster]

        with st.expander(
            f"Cluster {cluster} — {segment_names[cluster]}"
        ):

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Customers",
                int(row["Customer_Count"])
            )

            c2.metric(
                "Avg Income",
                f"₹{row['Annual_Income']:,.0f}"
            )

            c3.metric(
                "Spending Score",
                f"{row['Spending_Score']:.1f}"
            )

            c4.metric(
                "Purchase Frequency",
                f"{row['Purchase_Frequency']:.1f}"
            )


with tab2:

    st.subheader(
        "Annual Income vs Spending Score"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for cluster in sorted(
        df["Cluster"].unique()
    ):

        cluster_data = df[
            df["Cluster"] == cluster
        ]

        ax.scatter(
            cluster_data["Annual_Income"],
            cluster_data["Spending_Score"],
            s=70,
            alpha=0.75,
            label=(
                f"Cluster {cluster} - "
                f"{segment_names[cluster]}"
            )
        )

    ax.set_title(
        "Customer Segmentation"
    )

    ax.set_xlabel(
        "Annual Income (₹)"
    )

    ax.set_ylabel(
        "Spending Score (1-100)"
    )

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    ax.grid(
        alpha=0.25
    )

    st.pyplot(fig)

    st.subheader(
        "Age vs Spending Score"
    )

    fig2, ax2 = plt.subplots(
        figsize=(10, 6)
    )

    for cluster in sorted(
        df["Cluster"].unique()
    ):

        cluster_data = df[
            df["Cluster"] == cluster
        ]

        ax2.scatter(
            cluster_data["Age"],
            cluster_data["Spending_Score"],
            s=70,
            alpha=0.75,
            label=f"Cluster {cluster}"
        )

    ax2.set_title(
        "Age vs Spending Score"
    )

    ax2.set_xlabel(
        "Age"
    )

    ax2.set_ylabel(
        "Spending Score"
    )

    ax2.legend()

    ax2.grid(
        alpha=0.25
    )

    st.pyplot(fig2)

    st.subheader(
        "Customer Count by Cluster"
    )

    counts = (
        df["Cluster"]
        .value_counts()
        .sort_index()
    )

    fig3, ax3 = plt.subplots(
        figsize=(8, 5)
    )

    ax3.bar(
        counts.index.astype(str),
        counts.values
    )

    ax3.set_title(
        "Customers in Each Cluster"
    )

    ax3.set_xlabel(
        "Cluster"
    )

    ax3.set_ylabel(
        "Customer Count"
    )

    st.pyplot(fig3)


with tab3:

    st.subheader(
        "Elbow Method"
    )

    wcss = []

    k_range = range(
        2,
        9
    )

    for test_k in k_range:

        test_model = KMeans(
            n_clusters=test_k,
            random_state=42,
            n_init=10
        )

        test_model.fit(
            X_scaled
        )

        wcss.append(
            test_model.inertia_
        )

    fig4, ax4 = plt.subplots(
        figsize=(9, 6)
    )

    ax4.plot(
        list(k_range),
        wcss,
        marker="o"
    )

    ax4.axvline(
        k,
        linestyle="--",
        label=f"Selected K = {k}"
    )

    ax4.set_title(
        "Elbow Method for Optimal K"
    )

    ax4.set_xlabel(
        "Number of Clusters"
    )

    ax4.set_ylabel(
        "WCSS"
    )

    ax4.legend()

    ax4.grid(
        alpha=0.25
    )

    st.pyplot(fig4)

    elbow_table = pd.DataFrame({
        "K": list(k_range),
        "WCSS": [
            round(value, 2)
            for value in wcss
        ]
    })

    st.dataframe(
        elbow_table,
        use_container_width=True
    )


with tab4:

    st.subheader(
        "Segmented Customer Dataset"
    )

    st.dataframe(
        df,
        use_container_width=True,
        height=500
    )

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Segmented Customers CSV",
        data=csv_data,
        file_name="segmented_customers.csv",
        mime="text/csv"
    )


with tab5:

    st.subheader(
        "💡 Business Insights"
    )

    for cluster in summary.index:

        row = summary.loc[cluster]
        name = segment_names[cluster]

        if "High-Value" in name:

            recommendation = (
                "Focus on customer retention, VIP rewards, "
                "premium services, and personalized offers."
            )

        elif "Premium" in name:

            recommendation = (
                "Promote premium products, exclusive offers, "
                "and personalized recommendations."
            )

        elif "Potential" in name:

            recommendation = (
                "Use discounts, loyalty programs, and "
                "cross-selling to increase spending."
            )

        elif "Budget" in name:

            recommendation = (
                "Offer affordable products, value bundles, "
                "and targeted discounts."
            )

        elif "Affluent" in name:

            recommendation = (
                "Use personalized campaigns to increase "
                "engagement and purchase frequency."
            )

        else:

            recommendation = (
                "Use targeted promotions and monitor changes "
                "in spending behavior."
            )

        st.markdown(
            f"### Cluster {cluster}: {name}"
        )

        st.write(
            f"**Customers:** "
            f"{int(row['Customer_Count'])}"
        )

        st.write(
            f"**Average income:** "
            f"₹{row['Annual_Income']:,.0f}"
        )

        st.write(
            f"**Average spending score:** "
            f"{row['Spending_Score']:.1f}"
        )

        st.write(
            f"**Recommended strategy:** "
            f"{recommendation}"
        )

        st.divider()



st.caption(
    "Customer Segmentation using K-Means Clustering | "
    "Python • Pandas • NumPy • Scikit-learn • Matplotlib • Streamlit"
)
