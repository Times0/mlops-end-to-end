import requests
from PIL import Image, ImageDraw

image_path = "yolotest2.JPG"#"yolotest.jpeg"
image = Image.open(image_path)

endpoint_url = "https://schlop-02f1e6f8.mt-guc1.bentoml.ai/predict"

with open(image_path, "rb") as f:
    files = {"image": ("image.jpg", f, "image/jpeg")}
    response = requests.post(endpoint_url, files=files)

if response.status_code != 200:
    print(f"Error: {response.status_code}, {response.text}")
    exit()

result = response.json()
print(result)

def plot_results(image, result):
    """
    plots bounding boxes around the detected objects
    """
    draw = ImageDraw.Draw(image)
    for box in result["boxes"]:
        xyxy = box["xyxy"]  # [x1, y1, x2, y2]
        draw.rectangle(xyxy, outline="red", width=3) 
        draw.text((xyxy[0], xyxy[1]), box["class_name"], fill="red")
    image.show()

plot_results(image, result)