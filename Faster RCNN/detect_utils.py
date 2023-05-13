import torchvision.transforms as transforms
import cv2
import numpy
import numpy as np
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

# define the torchvision image transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

def predict(image, model, device, detection_threshold):
    # transform the image to tensor
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    outputs = model(image) # get the predictions on the image
    # print the results individually
    print(f"BOXES: {outputs[0]['boxes']}")
    print(f"LABELS: {outputs[0]['labels']}")
    # get all the predicited class names
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    # Threshold can be optimized for each problem
    threshold=0.5
    pred_scores = pred_scores >= threshold
    print(pred_scores)
    print(f"SCORES: {outputs[0]['scores']}")
    #x = [0,0.2,0.4,0.6,1.0]
    #y_values=[0,0.3,0.5,0.7,1.0]
    #y_axis=random.sample(y_values,5)
    # naming the x axis
    #plt.xlabel('evaluation metrics')
    # naming the y axis
    #plt.ylabel('confidence score')
    # giving a title to my graph
    #plt.title('Graph for object detection')
    #plt.plot(x,y_axis)
    # function to show the plot
    #plt.show()
    return boxes, pred_classes, outputs[0]['labels']

def draw_boxes(boxes, classes, labels, image):
    # read the image with OpenCV
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType=cv2.LINE_AA)
    
    return image
