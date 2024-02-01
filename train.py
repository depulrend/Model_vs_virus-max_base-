import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import joblib

# Загрузка данных для обучения
train_data = pd.read_csv('train.tsv', sep='\t')

# Создание модели
model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))

# Обучение модели
model.fit(train_data['libs'], train_data['is_virus'])

# Сохранение модели
joblib.dump(model, 'model.joblib')