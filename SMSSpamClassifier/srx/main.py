import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

def prep_data():
    df = pd.read_csv('data/spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'msg']

    X = df['msg']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train.to_csv('data/X_train.csv', index=False, header=False)
    X_test.to_csv('data/X_test.csv', index=False, header=False)
    y_train.to_csv('data/y_train.csv', index=False, header=False)
    y_test.to_csv('data/y_test.csv', index=False, header=False)

def train():
    X_train = pd.read_csv('data/X_train.csv', header=None)[0]
    X_test = pd.read_csv('data/X_test.csv', header=None)[0]
    y_train = pd.read_csv('data/y_train.csv', header=None)[0]
    y_test = pd.read_csv('data/y_test.csv', header=None)[0]

    vec = TfidfVectorizer(stop_words='english')
    models = {
        'NB': Pipeline([('vec', vec), ('clf', MultinomialNB())]),
        'LR': Pipeline([('vec', vec), ('clf', LogisticRegression(max_iter=1000))]),
        'SVC': Pipeline([('vec', vec), ('clf', SVC())])
    }

    best_model = None
    best_f1 = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, pos_label='spam')

        if f1 > best_f1:
            best_f1 = f1
            best_model = model

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label='spam')
        rec = recall_score(y_test, y_pred, pos_label='spam')

        with open('output/results.txt', 'a') as f:
            f.write(f'{name}:\n')
            f.write(f'Accuracy: {acc:.2f}\n')
            f.write(f'Precision: {prec:.2f}\n')
            f.write(f'Recall: {rec:.2f}\n')
            f.write(f'F1 Score: {f1:.2f}\n\n')

    joblib.dump(best_model, 'output/model/best_model.pkl')

def eval_model():
    X_test = pd.read_csv('data/X_test.csv', header=None)[0]
    y_test = pd.read_csv('data/y_test.csv', header=None)[0]

    model = joblib.load('output/model/best_model.pkl')
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='spam')
    rec = recall_score(y_test, y_pred, pos_label='spam')
    f1 = f1_score(y_test, y_pred, pos_label='spam')

    with open('output/results.txt', 'a') as f:
        f.write('Best Model Evaluation:\n')
        f.write(f'Accuracy: {acc:.2f}\n')
        f.write(f'Precision: {prec:.2f}\n')
        f.write(f'Recall: {rec:.2f}\n')
        f.write(f'F1 Score: {f1:.2f}\n')

if __name__ == "__main__":
    if not os.path.exists('output/model'):
        os.makedirs('output/model')
    if not os.path.exists('output'):
        os.makedirs('output')

    with open('output/results.txt', 'w') as f:
        f.write('')

    prep_data()
    train()
    eval_model()

    print("Done!")
