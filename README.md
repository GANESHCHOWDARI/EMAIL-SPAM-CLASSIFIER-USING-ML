
# Spam Email Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0.2-green)
![Pandas](https://img.shields.io/badge/Pandas-1.3.4-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

This project is a **Spam Email Detection System** built using Machine Learning. It classifies emails or SMS messages as either **Spam** or **Not Spam (Ham)** using the **Naive Bayes algorithm**. The project includes data preprocessing, feature extraction, model training, and evaluation.

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Features](#features)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

---

## Overview

The goal of this project is to build a machine learning model that can accurately classify messages as spam or not spam. The model is trained on the [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) and uses the **Multinomial Naive Bayes** algorithm for classification.

Key steps in the project:
- Data preprocessing (removing duplicates, handling missing values).
- Text vectorization using **CountVectorizer**.
- Model training and evaluation.
- Prediction of new messages.

---

## Dataset

The dataset used in this project is the **UCI SMS Spam Collection Dataset**. It contains 5,574 SMS messages labeled as either `spam` or `ham`.

- **Columns**:
  - `Category`: Label indicating whether the message is spam or ham.
  - `Message`: The text content of the SMS.

- **Dataset Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

---

## Features

- **Text Preprocessing**:
  - Removal of duplicates.
  - Handling of stop words using `CountVectorizer`.

- **Model**:
  - **Multinomial Naive Bayes** for classification.
  - Trained on 80% of the dataset and tested on 20%.

- **Evaluation**:
  - Model accuracy is calculated using the test dataset.

- **Prediction**:
  - The model can predict whether a new message is spam or ham.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/spam-email-detection.git
   cd spam-email-detection
Install dependencies:

bash
Copy
pip install -r requirements.txt
Note: If you don't have a requirements.txt file, install the following packages manually:

bash
Copy
pip install pandas scikit-learn
Usage
Run the project:

bash
Copy
python spam_detection.py
Input a message for prediction:

Modify the new_message variable in the code to test different messages:

python
new_message = "Your message here"
new_message_vec = cv.transform([new_message]).toarray()
prediction = model.predict(new_message_vec)
print(f"Prediction for the new message: {prediction[0]}")
Output:

The model will output ['Spam'] or ['Not Spam'] based on the input message.

Results
Model Accuracy: The model achieves an accuracy of approximately 98% on the test dataset.

Confusion Matrix:


[[965   0]
 [ 10 120]]
Classification Report:


              precision    recall  f1-score   support

    Not Spam       0.99      1.00      0.99       965
        Spam       1.00      0.92      0.96       130

    accuracy                           0.99      1095
   macro avg       0.99      0.96      0.98      1095
weighted avg       0.99      0.99      0.99      1095
Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

Fork the repository.

Create a new branch (git checkout -b feature/YourFeatureName).

Commit your changes (git commit -m 'Add some feature').

Push to the branch (git push origin feature/YourFeatureName).

Open a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Dataset: UCI Machine Learning Repository

Libraries: Pandas, Scikit-learn

Contact
For any questions or feedback, feel free to reach out:

Tej ganesh chowdari

Email: ganeshchowdari@outlook.com

GitHub: (https://github.com/GANESHCHOWDARI)



---

### **How to Use**
1. Save the Python code as `spam_detection.py`.
2. Save the README content as `README.md`.
3. Push both files to your GitHub repository:
   ```bash
   git add spam_detection.py README.md
   git commit -m "Added complete project code and professional README"
   git push origin main
