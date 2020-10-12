from GradCAM import GradCam
from util import preprocess_image, show_cam_on_image, make_transforms
import torch
import os
import cv2
from PIL import Image
import numpy as np
from torchsummary import summary
import argparse

def predict(model, test_image, print_class=False):

    chosen_transforms = make_transforms()
    transform = chosen_transforms['data/val']
    test_image_tensor = transform(test_image)
    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 227, 227).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 227, 227)

    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(1, dim=1)
        class_name = idx_to_class[topclass.cpu().numpy()[0][0]]
        if print_class:
            print("Output class :  ", class_name)
    return class_name


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model',
                        help='Input image path')
    parser.add_argument('--save', type=str, default='GradCAM',
                        help='Saved folder name')
    args = parser.parse_args()

    return args


if __name__ == '__main__':


    args = get_args()
    cwd = os.getcwd()
    model_name = args.model + '.pt'
    SAVED_NAME = args.save
    PATH_model = os.path.join(cwd, model_name)

    model = torch.load(PATH_model)
    # summary(model, (3, 227, 227))
    grad_cam = GradCam(model=model, feature_module=model.layer4, target_layer_names=["2"], use_cuda=True)
    test_img_dir = os.path.join(cwd, 'test_image')
    GradCAM_dir = os.path.join(cwd, SAVED_NAME)
    if not os.path.isdir(GradCAM_dir):
        os.makedirs(GradCAM_dir)

    idx_to_class = {}
    class_names = os.listdir(os.path.join(cwd, 'data', 'train'))
    for idx, label in enumerate(class_names):
        idx_to_class[idx] = label
    print(class_names)
    print(idx_to_class)
    cnt = 1
    for image in os.listdir(test_img_dir):
        img = cv2.imread(os.path.join(test_img_dir, image), 1)
        # 예측한 클래스를 파일의 제목으로 지정
        predicted_class = predict(model, Image.fromarray(img))
        # predicted_class = "no_class"

        img = np.float32(cv2.resize(img, (224, 224))) / 255
        input = preprocess_image(img)
        target_index = None
        mask = grad_cam(input, target_index)
        cam = show_cam_on_image(img, mask)
        saved_name = os.path.join(GradCAM_dir, str(cnt) + "_" + predicted_class + ".jpg")
        cv2.imwrite(saved_name, np.uint8(255 * cam))
        cnt += 1
