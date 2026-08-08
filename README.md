# Customer Segmentation using K-Means Clustering

A complete machine learning project that segments customers using the **K-Means Clustering Algorithm**, with an interactive **Streamlit web application** ready for GitHub deployment.

## 🚀 Live Streamlit Deployment

Deploy `app.py` using Streamlit Community Cloud.

**Main file path:**

```text
app.py
```

After deployment, add the generated public Streamlit URL here:

```text
https://YOUR-APP-NAME.streamlit.app
```

## 📌 Project Objective

The project groups customers with similar characteristics using:

- Age
- Annual Income
- Spending Score
- Purchase Frequency

The resulting segments can help businesses create targeted marketing strategies, personalized offers, and data-driven decisions.

## 🧠 Machine Learning Method

The project uses:

1. Data collection
2. Data preprocessing
3. Feature selection
4. Feature scaling
5. Elbow Method
6. K-Means clustering
7. Cluster analysis
8. Visualization
9. Business recommendations

## 🌐 Streamlit Application

`app.py` is a self-contained Streamlit application.

It includes:

- Built-in 300-customer demonstration dataset
- Optional CSV upload
- Adjustable K value
- K-Means clustering
- Silhouette Score
- Elbow Method
- Cluster summary
- Customer visualizations
- Automatic customer segment names
- Business recommendations
- Downloadable segmented CSV

### Expected CSV format

If you upload your own dataset, use these columns:

```text
Customer_ID
Age
Gender
Annual_Income
Spending_Score
Purchase_Frequency
```

## 📂 Repository Structure

```text
customer-segmentation/
│
├── app.py
├── requirements.txt
├── README.md
├── PROJECT_SYNOPSIS.md
├── LICENSE
├── .gitignore
│
├── data/
│   └── customers.csv
│
├── notebooks/
│   └── customer_segmentation.ipynb
│
├── src/
│   └── customer_segmentation.py
│
└── outputs/
    ├── elbow_method.png
    ├── customer_clusters.png
    ├── cluster_summary.csv
    └── segmented_customers.csv
```

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Jupyter Notebook

## ▶️ Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 📊 Expected Output

The application displays:

- Number of customers
- Number of clusters
- Average income
- Silhouette score
- Cluster characteristics
- Income vs Spending Score graph
- Age vs Spending Score graph
- Elbow Method graph
- Customer segmentation table
- Business recommendations

## 💡 Example Customer Segments

Depending on the dataset, clusters may represent:

- High-Value Loyal Customers
- Premium Customers
- Potential Growth Customers
- Budget Customers
- Affluent Low-Engagement Customers
- Moderate Customers

Cluster numbers themselves have no fixed business meaning; they should be interpreted from their characteristics.

## ⚠️ Important

The included customer dataset is synthetic and intended for academic/project demonstration. A real business dataset can be uploaded through the Streamlit application.

## 👨‍💻 Author

B.Tech Student | Aspiring Data Analyst
