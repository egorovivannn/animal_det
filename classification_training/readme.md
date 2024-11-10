Для обучения классификатора

вести следующие команды:
```
  python3.11 -m venv myvenv
  
  source myvenv/bin/activate
  
  pip install -r requirements.txt
```

Подготовка annotation_split_clusters.csv файла описано в ноутбуке prepare_df.ipynb

Для обучения модели необходимо запустить команду:

CUDA_VISIBLE_DEVICES=0 python train_vits.py  --cfg ./configs/configs_vit_small.yaml
CUDA_VISIBLE_DEVICES=0 python train_vits.py  --cfg ./configs/configs_vit_giant.yaml

Для запуска инференса используется скрипт process_folder.py. Пример использования:
CUDA_VISIBLE_DEVICES=0 python process_folder.py --folder_path ./train_data_minprirodi/images
