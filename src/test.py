from ultralytics import YOLO
import os

# Define model path
model_path = os.path.join(os.getcwd(), "tmp", "models", "best.pt")

# Check if the file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
else:
    try:
        # Load model
        model = YOLO(model_path)
        print("Model loaded successfully!")
        
        # Perform a dry-run with a dummy image (random noise)
        import numpy as np
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model.predict(dummy_image)
        
        print("Inference successful! Here are the results:")
        print(results[0])
    except Exception as e:
        print(f"Error loading model: {e}")
