import pandas as pd
import cv2
import glob
import os
from sklearn.model_selection import train_test_split
import shutil

DATA_PATH = '/home/ivan/animals_hack/train_data_minprirodi/'


df = pd.read_csv(f'{DATA_PATH}/annotation.csv')

for line in df.itertuples():
    idx = line.Index
    bbox = line.Bbox
    name = line.Name
    
    img = cv2.imread(f'{DATA_PATH}/images/{name}')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    x_center, y_center, w_bbox, h_bbox = [float(i) for i in bbox.split(',')]
        
    x_center = int(x_center*w)
    w_bbox = int(w_bbox*w)
    y_center = int(y_center*h)
    h_bbox = int(h_bbox*h)
    
    x1 = x_center - w_bbox // 2
    y1 = y_center - h_bbox // 2
    
    x2 = x1 + w_bbox
    y2 = y1 + h_bbox

    df.loc[idx, 'x_center'] = x_center
    df.loc[idx, 'w_bbox'] = w_bbox
    df.loc[idx, 'y_center'] = y_center
    df.loc[idx, 'h_bbox'] = h_bbox
    
    df.loc[idx, 'x1'] = x1
    df.loc[idx, 'y1'] = y1
    df.loc[idx, 'x2'] = x2
    df.loc[idx, 'y2'] = y2

df.to_csv(f'{DATA_PATH}df.csv', index=False)


names = df.Name.unique()
df['Bbox'] = df['Bbox'].apply(lambda x: x.replace(',', ' '))
df['Class'] = df['Class'].astype(str)

os.makedirs(f'{DATA_PATH}/labels/', exist_ok=True)
for name in names:
    temp = df[df['Name']==name][['Class', 'Bbox']].values.tolist()
    temp = '\n'.join([' '.join(i) for i in temp])
    with open(f'{DATA_PATH}labels/{name.replace("jpg", "txt")}', 'w') as f:
        f.writelines(temp)

for name in glob.glob(f'{DATA_PATH}images_empty/*'):
    with open(f'{DATA_PATH}labels/{os.path.basename(name).replace("jpg", "txt")}', 'w') as f:
        f.writelines('')

for old_path in glob.glob(f'{DATA_PATH}images_empty/*'):
    shutil.copy(old_path, f'{DATA_PATH}/images/{os.path.basename(old_path)}')


all_files = glob.glob(f'{DATA_PATH}/images/*')
train_paths, val_paths = train_test_split(all_files, test_size=0.1)

with open(f'{DATA_PATH}train.txt', 'w') as f:
    f.writelines('\n'.join(train_paths))

with open(f'{DATA_PATH}val.txt', 'w') as f:
    f.writelines('\n'.join(val_paths))