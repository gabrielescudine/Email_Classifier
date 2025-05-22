# Overview | E-mail Classifier
This is a classic Machine Learning problem that aims to use classification algorithms to filter emails as either spam or not. The model developed is simple, yet robust and highly efficient, achieving noteworthy performance metrics. The main goal of this project is to deepen the understanding of Machine Learning models and algorithms, as well as to apply concepts such as exploratory data analysis, data visualization, and the development of a predictive model.

### Features
* **Email Classification**: Automatically classifies emails as Spam or Not Spam using machine learning algorithms.
* **Multinomial Naive Bayes Model**: Implements a simple yet robust and efficient classifier based on the Multinomial Naive Bayes algorithm.
* **Exploratory Data Analysis**: Includes detailed data exploration and visualizations to better understand the dataset.
* **Data Visualization**: Generates insightful plots using Matplotlib and Seaborn to support data analysis and model evaluation.
* **Model Evaluation**: Uses standard classification metrics such as accuracy, precision, recall, and confusion matrix for performance evaluation.
* **Easy Setup**: Clear instructions for installation and execution using a virtual environment and ```requirements.txt```.

### Data
The dataset used in this project was obtained from the Kaggle platform. You can access it at the following link: [Kaggle - Email Spam Classification Dataset](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv/data)


## Installation
1. Clone this repository to your local machine:
```bash
git clone https://github.com/gabrielescudine/Email_Classifier
cd Email_Classifier
```
2. Create a virtual environment and activate it:
```bash
python -m venv venv
venv\Scripts\activate # Windows
source venv/bin/activate # Linux/MacOS
```
3. Install the required dependencies using the requirements.txt:
```bash
pip install -r requirements.txt
```

## How to use
To run the ```spam_detection_model.py```, certify that you have installed all the necessary dependencies, go to the ```model/``` folder and execute the following command:
```bash
cd model/
python spam_detection_model.py
```
After completing all the necessary setup, the script should be ready to run.

## Project Structure
* ```data/``` — contains datasets used for training and testing
* ```model/``` — scripts for training and validating the model
* ```notebooks/``` — notebooks with exploratory data analysis and visualizations
* ```requirements.txt``` — list of required libraries

## Next Steps
- [⌛] Implement synthetic data generation to further validate the model's robustness.
- [⌛] Explore alternative classification algorithms such as **Support Vector Machines (SVM)** and **Random Forest**.
- [⌛] Add more visualizations to enhance the understanding and presentation of the model's results.

## License
This project is licensed under the MIT License — see the LICENSE file for details. </br>
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)