import cv2
import glob
import numpy as np
import time
import torch, torchvision


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_to_tensor(images, device):
    image_list = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        image_list.append(img)
    
    image_list = np.array(image_list)
    return torch.from_numpy(image_list).float().to(device)

def get_model(device):
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to(device)
    return model

def infer(tensor, model):
    since = time.time()
    outputs = model(tensor)
    print(f"Elapsed time: {time.time() - since} [sec]")
    return outputs

def sigmoid(x):
    return 1./(1. + np.exp(-0.1 * x))

def post_process(images, outputs, threshold=0.7):
    image_list = []
    for i, output in enumerate(outputs):
        bbs = output['boxes'].to(torch.int16).cpu().numpy()
        scores = output['scores'].to(torch.float16).cpu().detach().numpy()
        keypoints = output['keypoints'].to(torch.int16).cpu().numpy()
        keypoints_scores = output['keypoints_scores'].to(torch.float16).cpu().detach().numpy()

        img = images[i]
        for index, bb in enumerate(bbs):
            if scores[index] >= threshold:
                cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 2)

        for index, keypoint_list in enumerate(keypoints):
            if scores[index] < threshold:
                continue

            keypoint_score = keypoints_scores[index]        
            keypoint_score = sigmoid(keypoint_score)
            for id, keypoint in enumerate(keypoint_list):
                x = keypoint[0]
                y = keypoint[1]

                score = keypoint_score[id]

                cv2.circle(img, (x, y), 3, (0, 0, 255), 1)

        image_list.append(img)

    return image_list


if __name__ == '__main__':
    device = get_device()
    keypoint_rcnn_model = get_model(device)
    
    # load images
    images = glob.glob("./images/*.jpg")
    im = cv2.imread(images[0])

    tensor = image_to_tensor([im], device)
    outputs = infer(tensor, keypoint_rcnn_model)

    result_image_list = post_process([im], outputs, threshold=0.3)
    cv2.imshow("output", result_image_list[0])
    cv2.waitKey(0)
