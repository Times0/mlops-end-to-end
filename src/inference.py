import mlflow
from ultralytics import YOLO
from pathlib import Path
from rich.console import Console
import os

console = Console()

def setup_mlflow():
    """Setup MLflow connection"""
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Configure MLflow to use MinIO
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://localhost:9000"
    os.environ['AWS_ACCESS_KEY_ID'] = "minioadmin"
    os.environ['AWS_SECRET_ACCESS_KEY'] = "minioadmin"

def load_champion_model():
    """Load the current champion model from MLflow"""
    client = mlflow.MlflowClient()
    try:
        # Get the champion version
        champion_version = client.get_model_version_by_alias("YOLO-model", "Champion")
        console.print(f"[green]Loading champion model version: {champion_version.version}[/]")
        
        # Get the run ID
        run_id = champion_version.run_id
        
        # Load the model
        model = YOLO(f"runs:/{run_id}/model")
        return model
    
    except Exception as e:
        console.print(f"[red]Error loading champion model: {str(e)}[/]")
        return None

def predict(model, image_path):
    """
    Perform prediction on a single image
    Args:
        model: YOLO model
        image_path: Path to the image file
    Returns:
        results: YOLO prediction results
    """
    try:
        results = model.predict(
            source=image_path,
            conf=0.25,  # confidence threshold
            save=True   # save results
        )
        return results
    except Exception as e:
        console.print(f"[red]Error during prediction: {str(e)}[/]")
        return None

if __name__ == "__main__":
    # Setup MLflow
    setup_mlflow()
    
    # Load the champion model
    model = load_champion_model()
    
    if model:
        # Example: predict on a single image
        image_path = "data/THE-dataset/test/images/IMG_1131.png"
        console.print(f"[green]Performing prediction on: {image_path}[/]")
        
        results = predict(model, image_path)
        
        if results:
            # Print results summary
            for r in results:
                console.print(f"[green]Detected {len(r.boxes)} objects[/]")
                
                # Print detailed results for each detection
                for box in r.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    class_name = model.names[class_id]
                    console.print(f"[blue]Found {class_name} with confidence {confidence:.2f}[/]")
