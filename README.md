# Binary Bank Classifier Prediction & Visualization

This project analyzes a Kaggle dataset (https://www.kaggle.com/competitions/playground-series-s5e8/overview) of bank customers to predict the likelihood of them subscribing to a bank term deposit. It involves data cleaning, exploratory data analysis, training a Random Forest classification model, and deploying an interactive web application using Streamlit to visualize the results.

The primary goal is to build a reliable binary classification model and present the findings in an accessible, user-friendly interface.

---

## Features

* **Interactive Web App**: A user-friendly dashboard built with Streamlit to visualize data and model predictions.
* **Exploratory Data Analysis**: Analysis and visualization of customer demographics and banking behavior.
* **Machine Learning Model**: Utilizes a trained Random Forest Classifier to predict customer subscriptions.
* **Performance Metrics**: Displays model performance using a classification report and a confusion matrix.

---

## Technologies Used

* **Programming Language**: Python
* **Data Manipulation**: Pandas, NumPy
* **Machine Learning**: Scikit-learn
* **Web Framework**: Streamlit
* **Data Visualization**: Plotly

---

## Live website

https://bank-subscription-insights.streamlit.app/

---

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need to have Python 3.7+ and `pip` installed on your system.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/YourUsername/Your-Repo-Name.git](https://github.com/YourUsername/Your-Repo-Name.git)
    cd Your-Repo-Name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    # For Mac/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

---

## Usage

To launch the interactive web application, you must run the command from the **root directory** of the project.

```sh
streamlit run src/dashboard/app.py
```

---

## License

This project is licensed under the APACHE 2.0 License - see the LICENSE file for details.
