from sklearn.metrics import accuracy_score # Accuracy metrics
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import data as dt

def train_model(X_train, y_train):
    X_train, X_test, y_train, y_test = dt.load_and_split_data()

    pipelines = {
        'rf':make_pipeline(StandardScaler(), RandomForestClassifier())

    }
    fit_models = {}
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(X_train, y_train)
        fit_models[algo] = model

    for algo, model in fit_models.items():
        yhat = model.predict(X_test)
        print(algo, accuracy_score(y_test, yhat))

    with open('body_language.pkl', 'wb') as f:
        pickle.dump(fit_models['rf'], f)

    return model