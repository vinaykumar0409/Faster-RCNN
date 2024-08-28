from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import os
from werkzeug.utils import secure_filename
from utils import draw_bounding_boxes
import torchvision
from torchvision.ops import nms

# load a model pre-trained on COCO
PATH = 'faster-rcnn-v1.pth'

device = torch.device('cpu')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Model class must be defined somewhere
model = torch.load(PATH, map_location=device)
model = model.to(device)

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image)

    # Perform NMS on the CPU
    for idx, pred in enumerate(prediction):
        pred['boxes'] = pred['boxes'].cpu()
        pred['scores'] = pred['scores'].cpu()
        pred['labels'] = pred['labels'].cpu()

        keep = nms(pred['boxes'], pred['scores'], iou_threshold=0.95)
        prediction[idx]['boxes'] = prediction[idx]['boxes'][keep]
        prediction[idx]['scores'] = prediction[idx]['scores'][keep]
        prediction[idx]['labels'] = prediction[idx]['labels'][keep]

    return prediction


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Run the prediction
            prediction = predict_image(file_path)
            print(prediction)

            # Draw bounding boxes on the image
            result_img_path = draw_bounding_boxes(filename, prediction)

            return render_template('results.html', image_file=result_img_path)
    return render_template('prediction-page.html')


@app.route('/results')
def results():
    return render_template('results.html')


if __name__ == '__main__':
    app.run(debug=True)