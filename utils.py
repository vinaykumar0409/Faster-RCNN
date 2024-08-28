import os
from PIL import ImageDraw, Image

def draw_bounding_boxes(img_path, prediction):
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    boxes = prediction[0]['boxes'].cpu().numpy()  # Bounding boxes
    labels = prediction[0]['labels'].cpu().numpy()  # Class labels
    scores = prediction[0]['scores'].cpu().numpy()  # Confidence scores

    for box, label, score in zip(boxes, labels, scores):
        if score < 0.90 :
            continue
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
        text = f"pedestrain {score:.2f}"  # Class name and confidence score
        print(text)
        draw.text((box[0], box[1]), text, fill="red")

    # Create the result directory if it doesn't exist
    result_dir = "static/uploads/results/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # Create the result path
    result_img_path = os.path.join(result_dir, os.path.basename(img_path))

    return result_img_path[7:]