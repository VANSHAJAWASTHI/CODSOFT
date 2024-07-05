import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib

if __name__ == "__main__":
    train_data = pd.read_csv('data/train_data_processed.csv')
    
    X_train, X_valid, y_train, y_valid = train_test_split(train_data['processed_description'], train_data['Genre'], test_size=0.2, random_state=42)
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=2000)),
        ('clf', MultinomialNB())
    ])
    
    param_grid = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__alpha': [0.5, 1]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=2, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    
    y_valid_pred = grid_search.predict(X_valid)
    print(f"Validation Accuracy: {accuracy_score(y_valid, y_valid_pred)}")
    print(f"Validation Classification Report:\n {classification_report(y_valid, y_valid_pred)}")
    
    best_model = grid_search.best_estimator_
    best_model.fit(train_data['processed_description'], train_data['Genre'])
    joblib.dump(best_model, 'models/best_model.pkl')
