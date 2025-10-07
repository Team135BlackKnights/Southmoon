from ultralytics import YOLO

if __name__ == '__main__':  # This is required for Windows!
    # Load a pretrained YOLOv8 model
    '''model = YOLO('yolov8x.pt')

    # Train on your dataset
    print("Starting training...")
    results = model.train(
        data='C:/Users/grant/Documents/Southmoon/setup/robot-bumpers/data.yaml',
        epochs=100,
        imgsz=640,
        batch=14,
        device=0,
        name='roboflow_model',
        workers=0  # Set to 0 to avoid multiprocessing issues on Windows
    )

    print("Training complete! Now exporting to CoreML...")'''

    # Load the best model from training
    trained_model = YOLO('/Users/pennrobotics/Downloads/best.pt') 

    # Export to CoreML
    trained_model.export(
    format='coreml',
    nms=True
)

    print("Done! Your .mlpackage file is ready.")