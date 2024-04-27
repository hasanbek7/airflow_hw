import datetime
import json

import pandas as pd
import os
import glob
import dill

path = os.environ.get('PROJECT_PATH')
local = 'C:/Users/DataScience/airflow_hw'


def load_model():
    with open(f'{path}/data/models/cars_pipe.pkl', 'rb') as file:
        model = dill.load(file)
    return model

def transform_jsons():
    # Укажите путь к директории с JSON-файлами
    json_dir = f'{path}/data/test'
    json_pattern = os.path.join(json_dir, '*.json')
    file_list = glob.glob(json_pattern)

    model = load_model()

    dfs = []  # Список для хранения датафреймов

    for file in file_list:
        with open(file) as f:
            json_data = pd.json_normalize(json.loads(f.read()))  # Преобразование JSON в датафрейм
            dfs.append(json_data)  # Добавление датафрейма в список

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['pred'] = model.predict(combined_df)
    combined_df[['id', 'pred']].to_csv(f'{path}/data/predictions/pred_{datetime.datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)



def predict():
    transform_jsons()
    print('Finish!')



if __name__ == '__main__':
    predict()
