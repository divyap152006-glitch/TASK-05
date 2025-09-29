# TASK-05
# Heart Disease Prediction using Decision Tree & Random Forest

This project predicts **heart disease** using machine learning models
such as **Decision Trees** and **Random Forests**. It performs **data
analysis**, trains models, evaluates performance, and visualizes results
with Python libraries.

------------------------------------------------------------------------

## 📂 Project Structure

    heart-disease-prediction/
    │
    ├── heart.csv               # Dataset file (replace with your path)
    ├── heart_disease.py        # Python script with full code
    ├── README.md               # Project documentation (this file)

------------------------------------------------------------------------

## 🚀 Features

-   Load and explore the Heart Disease dataset\
-   Train & evaluate **Decision Tree** classifier\
-   Control overfitting using **max_depth**\
-   Train **Random Forest** classifier\
-   Visualize confusion matrix and feature importance\
-   Perform **cross-validation** for accuracy estimation

------------------------------------------------------------------------

## 🛠️ Installation & Setup

1.  **Clone the repository**

    ``` bash
    git clone https://github.com/your-username/heart-disease-prediction.git
    cd heart-disease-prediction
    ```

2.  **Install dependencies**\
    Make sure you have Python installed, then run:

    ``` bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

3.  **Add dataset**\
    Place your `heart.csv` file in the project folder and update the
    file path in the code if needed.

------------------------------------------------------------------------

## 📊 Dataset

-   **Input Features:** Age, Sex, Cholesterol, Resting BP, Max HR, etc.\
-   **Target:**
    -   `1` → Heart Disease\
    -   `0` → No Heart Disease

------------------------------------------------------------------------

## 🧠 Models Implemented

### 1️⃣ Decision Tree Classifier

-   Trained on 80% data, tested on 20%.\
-   Visualized tree structure with **max_depth=3** for clarity.\
-   Controlled overfitting with `max_depth=4`.

### 2️⃣ Random Forest Classifier

-   Trained with 100 estimators (`n_estimators=100`).\
-   Plotted **feature importance** for better interpretability.

------------------------------------------------------------------------

## 📈 Evaluation Metrics

-   **Accuracy**\
-   **Classification Report** (Precision, Recall, F1-Score)\
-   **Confusion Matrix**\
-   **Cross-validation Accuracy**

------------------------------------------------------------------------

## 📊 Visualizations

-   **Decision Tree Structure** (limited depth)\
-   **Confusion Matrix Heatmap**\
-   **Feature Importance Bar Chart**

------------------------------------------------------------------------

## ▶️ Run the Code

Run the Python script in terminal or Jupyter Notebook:

``` bash
python heart_disease.py
```

------------------------------------------------------------------------

## 📌 Example Results

  Model                  Accuracy (Test)   Cross-Validation Accuracy
  ---------------------- ----------------- ---------------------------
  Decision Tree          0.78              0.76
  Pruned Decision Tree   0.80              0.78
  Random Forest          0.84              0.82

*Note: Results may vary depending on dataset split.*

------------------------------------------------------------------------

## 🖼️ Sample Plots

### Confusion Matrix

Shows correct & incorrect predictions visually.

### Feature Importance

Highlights which features influence predictions the most.

------------------------------------------------------------------------

## 📌 Future Improvements

-   Add **hyperparameter tuning** using GridSearchCV\
-   Use **other models** (Logistic Regression, SVM)\
-   Deploy model as a **web application** using Flask or Streamlit

------------------------------------------------------------------------

## 📜 License

This project is for **educational purposes only**.
