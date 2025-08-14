# 🌸 Iris Flower Classification 
This project predicts the species of an iris flower (**Setosa**, **Versicolor**, or **Virginica**) based on four physical measurements:  
- Sepal Length (cm)  
- Sepal Width (cm)  
- Petal Length (cm)  
- Petal Width (cm)  

It uses **Logistic Regression** and is deployed as an **interactive web app** with Streamlit.  
The dataset is the famous **Iris dataset** from Scikit-learn.

---

## 📂 Project Structure
iris-flower-classification/
│
├── app.py # Streamlit app for prediction
├── iris_classification.ipynb # Jupyter notebook (EDA + training)
├── requirements.txt # Dependencies for the project
├── dataset_info.txt # Information about the Iris dataset
├── README.md # Project documentation
└── LICENSE # License file (MIT recommended)

yaml
Copy
Edit

---

## 🚀 Live Demo
 **Streamlit Cloud:** [https://codesoft-task---iris-flower-classification-2wvwm5uvmv4towfnkui.streamlit.app/]
---

## 📊 Dataset Details
- **Samples:** 150
- **Features:** 4 (all numeric, in cm)
- **Classes:**  
  - Setosa (50 samples)  
  - Versicolor (50 samples)  
  - Virginica (50 samples)  
- **Source:** Built-in Scikit-learn dataset

---

## 🛠️ Tech Stack
- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **Streamlit**
- **Matplotlib, Seaborn** (for visualization in notebook)

---

## ⚙️ Installation & Setup
```bash
# Clone the repository
git clone https://github.com/YourUsername/iris-flower-classification.git
cd iris-flower-classification

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app locally
streamlit run app.py
📈 Model Performance
The Logistic Regression model achieves 96–100% accuracy depending on the train-test split.
It was trained on scaled features for optimal performance.
-----

## 📌 Usage
Open the app in your browser.

Adjust the sliders for sepal and petal measurements.

Click Predict to see the predicted species.

