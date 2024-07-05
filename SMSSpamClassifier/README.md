# Spam SMS Classifier

## Introduction

Hi! I’m **Vanshaj Awasthi**, a BTech CSE 2nd Year student. This project builds an AI model to classify SMS messages as **spam** or **ham** (legitimate). We use machine learning techniques to detect spam messages.

## Project Structure

Here’s the project layout:

spam-sms-classifier/
│
├── data/
│ └── spam.csv
│
├── src/
│ └── main.py
│
├── output/
│ ├── model/
│ │ ├── best_model.pkl
│ ├── results.txt
│
├── README.md
├── requirements.txt
└── .gitignore


## Dataset

`spam.csv` contains SMS messages with:
- **`v1`**: Label (spam or ham)
- **`v2`**: Message content

## Installation

Clone the repo and install packages:

```bash
git clone https://github.com/VANSHAJAWASTHI/spam-sms-classifier.git
cd spam-sms-classifier
pip install -r requirements.txt

Contributing
Feel free to contribute by opening issues or PRs. Suggestions and improvements are welcome!

Contact
Reach me at awasthivanshaj@gmail.com