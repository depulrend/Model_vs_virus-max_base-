import pandas as pd
import joblib

# Загрузка данных для предсказания
test_data = pd.read_csv('test.tsv', sep='\t')

# Загрузка обученной модели
model = joblib.load('model.joblib')

# Предсказание на проверочной выборке
predictions = model.predict(test_data['libs'])

# Запись результатов предсказания и объяснений в файлы
with open('prediction.txt', 'w') as pred_file, open('explain.txt', 'w') as expl_file:
    pred_file.write("prediction\n")
    for i, prediction in enumerate(predictions):
        pred_file.write(str(prediction) + "\n")
        if prediction == 1:
            expl_file.write(f"File {test_data.iloc[i]['libs']} is malicious.\n")
            expl_file.write("Explanation: Critical libraries detected.\n")
            expl_file.write("\n")