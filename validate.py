# validate.py
import pandas as pd
import joblib
from sklearn.metrics import classification_report


# Загрузка данных для валидации
val_data = pd.read_csv('val.tsv', sep='\t')

# Загрузка обученной модели
model = joblib.load('model.joblib')

# Валидация модели на валидационной выборке
predictions = model.predict(val_data['libs'])

# Расчет метрик качества
report = classification_report(val_data['is_virus'], predictions, output_dict=True)

# Запись результатов в файл
with open('validation.txt', 'w') as f:
    f.write(f"True positive: {report['1']['precision']}\n")
    f.write(f"False positive: {report['0']['precision']}\n")
    f.write(f"False negative: {report['1']['recall']}\n")
    f.write(f"True negative: {report['0']['recall']}\n")
    f.write(f"Accuracy: {report['accuracy']}\n")
    f.write(f"Precision: {report['macro avg']['precision']}\n")
    f.write(f"Recall: {report['macro avg']['recall']}\n")
    f.write(f"F1: {report['macro avg']['f1-score']}\n")
