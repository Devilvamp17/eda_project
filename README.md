# 📊 Amount Prediction Dataset - Exploratory Data Analysis (EDA)

This repository contains a detailed EDA on a retail dataset with the goal of understanding patterns in customer purchase behavior, especially focusing on the **`Amount`** spent.

## 📁 Project Structure

```
├── data.csv                   # Raw dataset
├── eda_script.py             # EDA implementation code
├── results/                  # Folder containing all saved plots
│   ├── gender_countplot.png
│   ├── amount_by_gender.png
│   ├── correlation_matrix.png
│   ├── zero_value_count.png
│   ├── original_distributions.png
│   ├── scaled_distributions.png
│   ├── feature_importance.png
│   ├── target_distribution.png
│   └── *.png (boxplots for categorical vs Amount)
├── eda_amount_prediction_report.txt  # Textual EDA summary
└── README.md
```

## 📌 Objective

To analyze and visualize the structure of the dataset in order to:
- Understand distributions and correlations.
- Identify key drivers of the `Amount` variable.
- Prepare the data for modeling (via scaling and encoding).

---

## 📑 Dataset Description

| Column            | Description                        |
|-------------------|------------------------------------|
| User_ID           | Unique customer ID                 |
| Cust_name         | Name of the customer               |
| Product_ID        | ID of the product purchased        |
| Gender            | Gender of the customer             |
| Age Group         | Age range category                 |
| Age               | Exact age                          |
| Marital_Status    | 0 = Unmarried, 1 = Married         |
| State             | State of residence                 |
| Zone              | Regional zone                      |
| Occupation        | Customer's occupation              |
| Product_Category  | Category of product                |
| Orders            | Number of orders                   |
| Amount            | Total amount spent (target)        |

---

## 🔍 Key Steps Performed

### ✅ Data Cleaning
- Dropped irrelevant columns: `Status`, `unnamed1`, `User_ID`
- Removed null values from `Amount`

### 📊 Visualizations
- Gender distribution bar plot
- Total amount by gender
- Correlation heatmap
- Count of zero values in numerical features
- Distribution plots before and after scaling
- Feature importance using Random Forest
- Target (`Amount`) distribution
- Boxplots of categorical features vs target

### ⚙️ Preprocessing
- Converted all numerical columns to `float`
- Applied scaling strategies based on skewness:
  - `StandardScaler`, `PowerTransformer`, `RobustScaler`
- Label encoded all object-type categorical features

### 🔥 Feature Importance
- Used `RandomForestRegressor` to evaluate feature contributions to `Amount`

---

## 📦 Output Highlights (Saved in `/results/`)

- `gender_countplot.png`  
- `correlation_matrix.png`  
- `original_distributions.png`, `scaled_distributions.png`  
- `feature_importance.png`  
- `target_distribution.png`  
- Boxplots like `Zone_vs_Amount_boxplot.png`, `Occupation_vs_Amount_boxplot.png`

---

## 🧪 Libraries Used

- `pandas`, `numpy` for data handling  
- `matplotlib`, `seaborn` for visualization  
- `scikit-learn` for preprocessing and modeling  
- `scipy.stats` for skew and kurtosis metrics  

---

## 💡 Next Steps

- Use the preprocessed dataset for building a regression model.
- Perform cross-validation and hyperparameter tuning.
- Explore feature engineering for further improvements.

---

## 📬 Author

*Arnav Goyal* – Passionate about data, AI/ML, and real-world problem solving.

---

## 📝 License

This project is for educational and demonstrative purposes.
