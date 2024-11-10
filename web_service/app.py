import streamlit as st
from PIL import Image
import io
import os
import glob

import albumentations as A
from processing import get_model
from ultralytics import YOLOWorld

from matplotlib import pyplot as plt
import torch
import cv2
import numpy as np
import pandas as pd
import json
import zipfile


image_size = 512
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
augs = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize(mean=mean, std=std, p=1)
])


model = YOLOWorld("./weights/best.pt")

model_classif = get_model(
    'vit_small_r26_s32_384.augreg_in21k_ft_in1k',
    'weights/best_model#384#vit_small_r26_s32_384_augreg_in21k_ft_in1k#split.pth',
    image_size=384
)



# Функция для обработки изображения и создания разметки
def process_image(file_name):
    img = plt.imread(file_name)
    r = model(img, verbose=False, half=True)[0]

    crops = []
    # bboxes_normalized = r.boxes.xywhn
    for box in r.boxes.xyxy:
        x1, y1, x2, y2 = box.detach().cpu().numpy().astype(int)
        crop = img[y1:y2, x1:x2]
    
        crop = augs(image=crop)['image']
        crop = torch.from_numpy(crop).permute(2, 0, 1)

        crops.append(crop)

    if len(crops) == 0:
        return {
        'bboxes': [], 
        'confs': [],
        'classes': []
    }

    images = torch.stack(crops)
    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        images = images.to('cuda')
        outputs = model_classif(torch.nn.functional.interpolate(images, size=(384, 384)))[0] 
        classes = (outputs.sigmoid() > 0.5).flatten().cpu().detach().numpy().astype('int').tolist()

    bboxes = r.boxes.xyxy.cpu().detach().numpy().astype('int').tolist()
    for idx in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[idx]
        h, w = y2-y1, x2-x1
        if h<min_bbox_h or w<min_bbox_w:
            classes[idx] = 0


    return {
        'bboxes': bboxes, 
        'confs': r.boxes.conf.cpu().detach().tolist(),
        'classes': classes
    }


def draw_bboxes(image, annotation):

    classes = annotation['classes']
    bboxes = annotation['bboxes']
    confs = annotation['confs']

    for ind in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[ind]

        if not classes[ind]:
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 5)
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 5)
        # cv2.putText(image, f"{classes[ind]}_{confs[ind]:.3}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,12,12), 2)"""  """

    return image


# Инициализируем session_state для сохранения загруженных файлов и аннотаций
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "annotations" not in st.session_state:
    st.session_state.annotations = {}

# Основной блок приложения
st.title("Сервис распознавания животных")

upload_method = st.radio("Выберите способ загрузки изображений:", ("Загрузить файлы вручную", "Указать путь к локальной папке"))

if upload_method=="Загрузить файлы вручную":
    st.session_state.FOLDER = False
    # Форма загрузки изображений
    uploaded_files = st.file_uploader(
        "Загрузите фото, доступен множественный выбор", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    # Обновляем список файлов в session_state после загрузки
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files

else:
    st.session_state.FOLDER = True
    # Ввод пути к папке
    folder_path = st.text_input("Введите полный путь к папке с изображениями")

col1, col2 = st.columns(2)
try:
    with col1:
        min_bbox_h = int(st.text_input("Минимальная высота бибокса", value="128").strip())
    with col2:
        min_bbox_w = int(st.text_input("Минимальная ширина бибокса", value="128").strip())
except:
    min_bbox_h, min_bbox_w = 128, 128




# Кнопка запуска обработки
if st.button("Запустить обработку"):
    
    if st.session_state.FOLDER:
        if not os.path.isdir(folder_path):
            st.write("Указанный путь не найден")
        else:
            
             # Очищаем предыдущие аннотации
            st.session_state.annotations = {}
            
            # Выполняем обработку для каждого изображения
            files = glob.glob(f'{folder_path}/*')
            name2path = {}
            for uploaded_file in files:
                name = os.path.basename(uploaded_file)
                name2path[name] = uploaded_file
                # Открываем изображение
                # image = Image.open(uploaded_file)
                
                # Обрабатываем изображение и создаем разметку
                annotation = process_image(uploaded_file)
                
                # Сохраняем разметку в session_state
                st.session_state.annotations[name] = annotation
            st.session_state.name2path = name2path

    elif st.session_state.uploaded_files:
        # Очищаем предыдущие аннотации
        st.session_state.annotations = {}
        
        # Выполняем обработку для каждого изображения
        for uploaded_file in st.session_state.uploaded_files:
            # Открываем изображение
            
            # Обрабатываем изображение и создаем разметку
            annotation = process_image(uploaded_file)

            
            # Сохраняем разметку в session_state
            st.session_state.annotations[uploaded_file.name] = annotation

# Если есть обработанные изображения, отображаем их и аннотации
if st.session_state.annotations:
    st.write("Результаты обработки:")


    info = {
        'positive': 0,
        'negative': 0,
        'empty': 0,
        'all': 0
    }
    
    for file_name, annotation in st.session_state.annotations.items():
        pos = sum(annotation['classes'])
        neg = len(annotation['classes']) - pos
        info['positive'] += pos
        info['negative'] += neg
        info['empty'] += 1 if not annotation['classes'] else 0

    info['all'] = info['positive']+info['negative']+info['empty']

    if any(info.values()):
        results_df = pd.DataFrame(list(info.items()), columns=["Класс", "Количество предсказаний"])
        st.table(results_df)


    show_method = st.radio("Выберите способ отображения результатов:", ("0. Не отображать", "1. Показать все", "2. Показать позитивные", "3. Показать негативные"))

    show_method = int(show_method[0])
    if show_method:
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w") as archive:
            for file_name, annotation in st.session_state.annotations.items():
                if show_method==2:
                    if 1 not in annotation['classes']:
                        continue
                if show_method==3:
                    if 0 not in annotation['classes']:
                        continue


                try:
                    if st.session_state.FOLDER:
                        image = Image.open(st.session_state.name2path[file_name])
                    else:
                        image = Image.open(next(f for f in st.session_state.uploaded_files if f.name == file_name))
                except Exception as e: 
                    print(e)
                    print('-'*20)
                    continue



                image = np.array(image)
                if annotation:
                    image = draw_bboxes(image, annotation)

                st.image(image, caption=file_name, use_container_width=True)
                


                annotation_file = json.dumps(annotation)
                # Кнопка для скачивания разметки
                st.download_button(
                    label="Скачать разметку",
                    data=annotation_file,
                    file_name=f"{os.path.splitext(file_name)[0]}.json",
                    mime="text/plain"
                )


                image_buffer = io.BytesIO()
                Image.fromarray(image).save(image_buffer, format="PNG")
                image_buffer.seek(0)
                archive.writestr(f"images/{file_name}", image_buffer.read())

                # annotation_buffer = io.BytesIO()
                # annotation_buffer.write(annotation_file)
                # annotation_buffer.seek(0)
                annotation_file_name = f"{os.path.splitext(file_name)[0]}.json"
                archive.writestr(f"labels/{annotation_file_name}", annotation_file)

        archive_buffer.seek(0)
        st.download_button(
            label="Скачать все результаты в виде архива",
            data=archive_buffer,
            file_name="results.zip",
            mime="application/zip"
        )