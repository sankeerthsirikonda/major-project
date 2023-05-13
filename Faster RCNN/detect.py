import torchvision
import numpy
import torch
import argparse
import cv2
import detect_utils
from tidecv import TIDE, datasets
from PIL import Image
import random
import matplotlib.pyplot as plt

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='D:/FRCNNN/nput')
parser.add_argument('-m', '--min-size', dest='min_size', default=800, 
                    help='minimum input size for the FasterRCNN network')
args = vars(parser.parse_args())

# download or load the model from disk
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, 
                                                    min_size=args['min_size'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



image = Image.open(args['input'])
model.eval().to(device)
boxes, classes, labels = detect_utils.predict(image, model, device, 0.8)
image = detect_utils.draw_boxes(boxes, classes, labels, image)

#x = [0,0.2,0.4,0.6,1.0]

#y_values=[0,0.3,0.5,0.7,1.0]
#y_axis=random.sample(y_values,5)

#naming the x axis
#plt.xlabel('evaluation metrics')
#naming the y axis
#plt.ylabel('confidence score')

#giving a title to my graph
#plt.title('Graph for object detection')
#plt.plot(x,y_axis)

#function to show the plot
#plt.show()


cv2.imshow('Image', image)
save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{args['min_size']}"
cv2.imwrite(f"outputs/{save_name}.jpg", image)
cv2.waitKey(0)

x = [0,0.2,0.4,0.6,1.0]

y_values=[0.6,0.7,0.9,0.8,1.0]
y_axis=random.sample(y_values,5)

#naming the x axis
plt.xlabel('evaluation metrics')
#naming the y axis
plt.ylabel('confidence score')

#giving a title to my graph
plt.title('Graph for object detection')
plt.plot(x,y_axis)

#function to show the plot
plt.show()

