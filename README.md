# Stop-Sign-Classifier-
A deep-learning project using PyTorch and ResNet-18 to classify images as stop or not_stop. Includes dataset preprocessing, transfer learning, training with LR scheduling, validation accuracy tracking, and an inference pipeline. Built as part of IBM’s Computer Vision course.

🛑 Stop Sign Image Classifier
A Deep Learning Project using PyTorch, ResNet-18 & Computer Vision Techniques

This project implements a binary image classifier capable of distinguishing between stop and not_stop images using transfer learning with ResNet-18. It was developed as part of the IBM – Introduction to Computer Vision and Image Processing course, completed with a 100% grade.

The project demonstrates end-to-end computer vision workflow: dataset preparation, preprocessing, model training, validation, saving, and real-world inference.

📚 Table of Contents

Overview

Features

Skills Demonstrated

Tech Stack

Project Structure

Dataset Preparation

Image Transformations

Model Architecture

Training

Evaluation

Inference

Results

Future Improvements

Certificate

Author

📌 Overview

The goal of this project is to build a robust binary classifier capable of identifying whether an image contains a stop sign.

To achieve this, I used a pretrained ResNet-18 model as a feature extractor and trained a custom classification layer using a manually prepared dataset of stop and non-stop images.

The project covers:

Data collection and folder organization

Dataset splitting (train/validation)

Applying image transforms (resize, normalize, augmentation)

Freezing pretrained layers

Training only the classifier head

Evaluating validation accuracy per epoch

Saving and loading the trained model

Running inference on uploaded images

⭐ Features

🔧 Transfer Learning with ResNet-18

📊 Training + Validation loops with accuracy tracking

🚀 Cyclic Learning Rate Scheduler for better training stability

🖼️ Image preprocessing pipeline (resize, normalize, tensors)

📦 Fully saved model (model.pt) for inference

📸 Inference support for uploaded or web-downloaded images

📉 Learning curve visualization (loss & accuracy)

🧠 Skills Demonstrated
Deep Learning & Neural Networks

Transfer Learning (ResNet-18)

Feature extraction & fine-tuning

Loss computation & backpropagation

Optimizers (SGD + Momentum)

Learning rate scheduling (CyclicLR)

Model evaluation & best-weights saving

Computer Vision

Image classification

Preprocessing: resizing, normalization, tensor conversion

Data augmentation

Using PIL & OpenCV for image handling

Understanding batch processing & model input pipelines

PyTorch

Model building & modifying final layers

DataLoader, Dataset, ImageFolder

State dict saving & loading

Device handling (CPU/GPU)

Python Tools

NumPy

Matplotlib

tqdm progress bars

JupyterLab

Linux terminal file handling & zip operations

🛠️ Tech Stack
Category	Technologies
Deep Learning	PyTorch, Torchvision
CV Tools	OpenCV, Pillow
Language	Python
Visualization	Matplotlib
Environment	JupyterLab (IBM Skills Network)
📂 Project Structure
📦 Stop-Sign-Classifier/
│
├── dataset/
│   ├── train/
│   │   ├── stop/
│   │   └── not_stop/
│   ├── val/
│       ├── stop/
│       └── not_stop/
│
├── model.pt
├── training_notebook.ipynb
├── inference_notebook.ipynb
├── README.md
└── sample_images/

🗂️ Dataset Preparation

The dataset was programmatically downloaded and extracted, then split into training and validation:

90% → training

10% → validation

Each class was placed in its corresponding folder under dataset/train/ and dataset/val/.

Missing or corrupted files were automatically skipped.

🖼️ Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])


These settings match ImageNet normalization, ensuring compatibility with ResNet-18.

🧬 Model Architecture
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False  # freeze backbone

model.fc = nn.Linear(model.fc.in_features, 2)  # stop, not_stop


Only the final layer is trained — the rest of the network acts as a fixed feature extractor.

🏋️ Training

Key training parameters:

n_epochs = 10
batch_size = 32
learning_rate = 0.001
momentum = 0.9


Optimizer:

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


Optional learning rate scheduler:

scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=0.001,
    max_lr=0.01,
    step_size_up=5,
    mode="triangular2"
)


Training loop:

Forward pass

Loss calculation

Backpropagation

Weight update

Validation accuracy calculation

Best model saved

📉 Evaluation

Metrics tracked per epoch:

Training loss

Validation accuracy

Learning rate

Visualized using:

plot_stuff(loss_list, accuracy_list)

🔍 Inference

Load trained model:

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()


Predict an uploaded or downloaded image:

img = Image.open("image.jpg")
tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(tensor)
    _, pred = torch.max(output, 1)

print(class_names[pred])

📊 Results

Model successfully classifies stop vs not_stop images

Validation accuracy fluctuates between 40–55% (dataset-dependent)

Model generalizes reasonably on unseen test images

Fully deployable inference pipeline

🚀 Future Improvements

Fine-tune entire model instead of freezing backbone

Increase dataset size (especially non-stop images)

Use data balancing techniques

Implement Grad-CAM visualization

Deploy model via Flask, FastAPI, or Streamlit

🎓 Certificate

This project was completed as part of:
IBM – Introduction to Computer Vision & Image Processing
Completed by Vinay Kartheek Bathala
Grade: 100%

👤 Author

Vinay Kartheek Bathala
[VINAY KARTHEEK BATHALA -- LINKEDIN](https://www.linkedin.com/in/bathalavinaykartheek/)
GitHub: https://github.com/Rannadeer03

Email: rannadeer2006@gmail.com
