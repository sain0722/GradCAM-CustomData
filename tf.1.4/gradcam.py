
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow.keras.backend as K
import numpy as np
import cv2
import os
#####################
# from tensorflow.keras.backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# graph = tf.get_default_graph()

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras!
# Otherwise, their weights will be unavilable in the threads after the session there has been set.
#####################


def load_image(path, target_size=(224, 224)):
    x = image.load_img(path, target_size=target_size)
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

class GradCAM:
    def __init__(self, model, activation_layer, class_idx):
        self.model = model
        self.activation_layer = activation_layer
        self.class_idx = class_idx
        self.tensor_function = self._get_gradcam_tensor_function()

    # get partial tensor graph of CNN model
    def _get_gradcam_tensor_function(self):
        model_input = self.model.input
        y_c = self.model.outputs[0].op.inputs[0][0, self.class_idx]
        A_k = self.model.get_layer(self.activation_layer).output

        tensor_function = K.function([model_input], [A_k, K.gradients(y_c, A_k)[0]])
        return tensor_function

    # generate Grad-CAM
    def generate(self, input_tensor):
        [conv_output, grad_val] = self.tensor_function([input_tensor])
        conv_output = conv_output[0]
        grad_val = grad_val[0]

        weights = np.mean(grad_val, axis=(0, 1))

        grad_cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
        for k, w in enumerate(weights):
            grad_cam += w * conv_output[:, :, k]

        grad_cam = np.maximum(grad_cam, 0)

        return grad_cam, weights


if __name__ == "__main__":

    img_width = 224
    img_height = 224

#    set_session(sess)

    model = load_model('./working/defect_model_keras.h5')
#    model.summary()
    activation_layer = "activation_48"


    cnt = 0
    correct_cnt = 0
    target_dir = "Defectimage_20200821/test/"
    DATA_DIR = "dataset/" + target_dir
    saved_dir = "result/"
#    img_path = "dataset/Defectimage_202000821/test/hhp/test_defect_000092_hhp.JPG"
#    img = load_image(img_path)
    for labels in os.listdir(DATA_DIR):
        print(labels)
        label_dir = os.path.join(DATA_DIR, labels)
        for img in os.listdir(label_dir):
            cnt += 1
            img_path = os.path.join(label_dir, img)
            img = load_image(img_path)                
            preds = model.predict(img)
            predicted_class = preds.argmax(axis=1)[0]
            for index, label in enumerate(["breakage", "hhp", "ms", "rec", "se", "wtw"]):
                #print(index, label)
                if index == predicted_class:
                    predicted_label = label
            print("predicted class : ", predicted_label)

            # Accuracy 
            if labels == predicted_label:
                correct_cnt += 1
            gardcam_generator = GradCAM(model, activation_layer, predicted_class)
            gradcam, gradcam_weight = gardcam_generator.generate(img)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_width, img_height))
            img_name_origin = saved_dir + str(cnt) + "_origin.jpg"
            cv2.imwrite(img_name_origin, img)

            gradcam = gradcam / gradcam.max()
            gradcam = gradcam * 255
            gradcam = cv2.resize(gradcam, (img_width, img_height))
            gradcam = np.uint8(gradcam)
            img_name_grad = saved_dir + str(cnt) + "_gradcam.jpg"

            cv_cam = cv2.applyColorMap(gradcam, cv2.COLORMAP_JET)
            fin = cv2.addWeighted(cv_cam, 0.5, img, 0.5, 0)
            cv2.imwrite(img_name_grad, fin)

    # validate
    print(model.get_layer(activation_layer).output_shape[1:3])
    Z = model.get_layer(activation_layer).output_shape[1] * model.get_layer(activation_layer).output_shape[2]
    print(gradcam_weight * Z)
    acc = float(correct_cnt / cnt)
    print("Accuracy: {}%".format(round(acc, 4) * 100))