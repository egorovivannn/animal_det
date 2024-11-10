# Руководство пользователя:

Для обучения детектора необходимо:


1) Ввести следующие команды:
```
  python3.11 -m venv myvenv
  
  source myvenv/bin/activate
  
  pip install -r requirements.txt
  
  Изменить переменную DATA_PATH в файле prepare_data на полный путь до обучающего датасета

  python prepare_data.py

  python train.py

```