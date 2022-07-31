import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from skimage import io



class RandomColorDistortion(tf.keras.layers.Layer):
    def __init__(self, contrast_range=(0.5, 1.5),
                 brightness_delta=(-0.2, 0.2), **kwargs):
        super(RandomColorDistortion, self).__init__(**kwargs)
        self.contrast_range = contrast_range
        self.brightness_delta = brightness_delta

    def get_config(self):
        config = super(RandomColorDistortion, self).get_config()
        config.update({
            'contrast_range': self.contrast_range,
            'brightness_delta': self.brightness_delta
        })
        return config

    def call(self, images, training=None):
        if not training:
            return images

        contrast = np.random.uniform(
            self.contrast_range[0], self.contrast_range[1])
        brightness = np.random.uniform(
            self.brightness_delta[0], self.brightness_delta[1])

        images = tf.image.adjust_contrast(images, contrast)
        images = tf.image.adjust_brightness(images, brightness)
        images = tf.clip_by_value(images, 0, 255)
        return images

def path_to_input_image(path, img_size=(224, 224) ):
    return img_to_array(load_img(path, target_size=img_size))

answer_list = [

        57.14, 50, 53.33, 55, 53.85, 58.62, 58.82, 51.11, 53.06, 55.77,
        66.67, 62.5, 66.67, 68.75, 60.87, 60.71, 68.97, 65.71, 65, 64.44,
        71.43, 71.43, 72.73, 73.33, 70, 73.91, 71.43, 76.67, 74.29, 72.5,
        86.96, 83.33, 88.89, 84.62, 82.35, 85, 80, 82.14, 86.67, 82.86,
        95.24, 90.91, 100, 91.67, 93.33, 94.44, 100, 95.83, 92.86, 96.67

    ]

def get_img_array(img_path, target_size):
    image1 = io.imread(img_path)
    io.imsave('temp.png', image1)
    # 載入檔案並調整影像尺寸(target_size)
    img = keras.utils.load_img(
        'temp.png', target_size=target_size)

    # 將 img 轉換成 shape (target_size, 3)、
    # 型別為 float32 的 numpy 陣列
    array = keras.utils.img_to_array(img)

    # 增加一個批次軸，現在 shape 變為 (1,target_size,3)
    array = np.expand_dims(array, axis=0)
    return array




def CNN_XGBRegression_model_all(
                                 CNN_model1=None,
                                 CNN_model2=None, CNN_model3=None, CNN_model4=None, CNN_model5=None,
                                 CNN_model6=None, CNN_model7=None, CNN_model8=None, CNN_model9=None, CNN_model10=None,
                                 CNN_model11=None, CNN_model12=None, CNN_model13=None, CNN_model14=None,
                                 XGBR_model=None, input_img_path=None):
    pre_first_input = get_img_array(input_img_path, target_size=(224, 224))
    pred_first1 = CNN_model1.predict(pre_first_input)[0][0]
    pred_first2 = CNN_model2.predict(pre_first_input)[0][0]
    pred_first3 = CNN_model3.predict(pre_first_input)[0][0]
    pred_first4 = CNN_model4.predict(pre_first_input)[0][0]
    pred_first5 = CNN_model5.predict(pre_first_input)[0][0]
    pred_first6 = CNN_model6.predict(pre_first_input)[0][0]
    pred_first7 = CNN_model7.predict(pre_first_input)[0][0]
    pred_first8 = CNN_model8.predict(pre_first_input)[0][0]
    pred_first9 = CNN_model9.predict(pre_first_input)[0][0]
    pred_first10 = CNN_model10.predict(pre_first_input)[0][0]
    pred_first11 = CNN_model11.predict(pre_first_input)[0][0]
    pred_first12 = CNN_model12.predict(pre_first_input)[0][0]
    pred_first13 = CNN_model13.predict(pre_first_input)[0][0]
    pred_first14 = CNN_model14.predict(pre_first_input)[0][0]
    pred_first = np.zeros((1, 14))
    pred_first[0][0] = pred_first1
    pred_first[0][1] = pred_first2
    pred_first[0][2] = pred_first3
    pred_first[0][3] = pred_first4
    pred_first[0][4] = pred_first5
    pred_first[0][5] = pred_first6
    pred_first[0][6] = pred_first7
    pred_first[0][7] = pred_first8
    pred_first[0][8] = pred_first9
    pred_first[0][9] = pred_first10
    pred_first[0][10] = pred_first11
    pred_first[0][11] = pred_first12
    pred_first[0][12] = pred_first13
    pred_first[0][13] = pred_first14
    pred_second = XGBR_model.predict(pred_first)
    pred_second = float(pred_second)
    return pred_second