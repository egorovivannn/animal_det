from ultralytics import YOLOWorld, YOLO
# Load a model
model = YOLOWorld("yolov8x-worldv2.pt")
model.train(
    data='cfg.yaml', 
    epochs=500, 
    batch=16, 
    verbose=True, 
    single_cls=True, 
    auto_augment='autoaugment', 
    iou=0.3,
    crop_fraction=0.5, 
    flipud=0.5, 
    mosaic=0.3, 
    copy_paste=0.1, 
    save_period=-1,
    lr0=0.0001,
    lrf=0.0001,
    optimizer='AdamW'
)