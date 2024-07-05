import pandas as pd
import joblib

if __name__ == "__main__":
    test_data = pd.read_csv('data/test_data_processed.csv')
    best_model = joblib.load('models/best_model.pkl')
    
    y_pred = best_model.predict(test_data['processed_description'])
    
    for idx, prediction in enumerate(y_pred):
        print(f"{test_data.iloc[idx]['ID']} ::: {test_data.iloc[idx]['Title']} ::: {prediction} ::: {test_data.iloc[idx]['Description']}")
