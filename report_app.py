import streamlit as st
import joblib
from functions_of_report_app import *

st.image(io.imread('transmon.png'))
st.write("""
***
""")

method = st.selectbox('Methods', [1, 2], index=0)
st.write("""
#### Description of different method:
**Method 1** : With higher bias but lower variance\n
**Method 2** : With higher variance but lower bias
""")

allow_range = st.slider('Allowable error range', 0.0, 30.0, 20.0, 0.5, )

if method==1:
    acc_list = [0.,    0.158, 0.28,  0.356, 0.41,  0.446, 0.49,  0.526, 0.552, 0.568, 0.584, 0.594,
                 0.598, 0.618, 0.624, 0.624, 0.63,  0.64,  0.648, 0.658, 0.658, 0.666, 0.68,  0.684,
                 0.692, 0.698, 0.706, 0.726, 0.738, 0.75,  0.762, 0.768, 0.776, 0.782, 0.784, 0.794,
                 0.802, 0.808, 0.812, 0.82,  0.828, 0.836, 0.838, 0.848, 0.854, 0.856, 0.86,  0.862,
                 0.866, 0.874, 0.876, 0.882, 0.886, 0.89,  0.898, 0.902, 0.902, 0.904, 0.906, 0.912,
                 0.92 ]

if method==2:
    acc_list = [0.,    0.05,  0.08,  0.13,  0.186, 0.22,  0.278, 0.328, 0.37,  0.392, 0.428, 0.456,
                 0.494, 0.502, 0.52,  0.546, 0.564, 0.596, 0.632, 0.662, 0.676, 0.686, 0.694, 0.702,
                 0.71,  0.72,  0.728, 0.734, 0.74,  0.756, 0.768, 0.778, 0.8,   0.808, 0.814, 0.814,
                 0.822, 0.83,  0.834, 0.846, 0.852, 0.86,  0.866, 0.87,  0.876, 0.878, 0.886, 0.89,
                 0.896, 0.898, 0.906, 0.912, 0.918, 0.92,  0.93,  0.936, 0.936, 0.936, 0.94,  0.942,
                 0.942]

def find_acc():
    ranges = 0.0
    for i in range(len(acc_list)):
        if allow_range == ranges:
            acc = acc_list[i]
            break
        ranges += 0.5

    return acc


acc = find_acc()
sigma = np.sqrt(acc * (1 - acc) / 500)
low_bound = acc - 3 * sigma
high_bound = acc + 3 * sigma
st.write(f'## Accuracy: {round(acc * 100, 2)} % ')
st.write(f'### Confidence level: {99.7} % ')
st.write(f'### Confidence interval of accuracy: {round(low_bound * 100, 2)}% ~ {round(high_bound * 100, 2)}%')
st.write(f'### Sampling error: {round(3 * sigma * 100, 2)} %')

st.write("""
***
""")
@st.cache(allow_output_mutation=True)
def load_model():
    model_list = []
    final_test_model = keras.models.load_model(
        "./models/2tuning_model_v3_xg_1.keras",
        custom_objects={'RandomColorDistortion': RandomColorDistortion}
    )
    model_list.append(final_test_model)
    final_test_sub_model1 = keras.models.load_model(
        "./models/try_cnn_xgboost_model_2_1.keras",
        custom_objects={'RandomColorDistortion': RandomColorDistortion}
    )
    model_list.append(final_test_sub_model1)
    final_test_sub_model2 = keras.models.load_model(
        "./models/try_cnn_xgboost_model_2_2.keras",
        custom_objects={'RandomColorDistortion': RandomColorDistortion}
    )
    model_list.append(final_test_sub_model2)
    final_test_sub_model3 = keras.models.load_model(
        "./models/try_cnn_xgboost_model_2_3.keras",
        custom_objects={'RandomColorDistortion': RandomColorDistortion}
    )
    model_list.append(final_test_sub_model3)
    final_test_sub_model4 = keras.models.load_model(
        "./models/first_learn_v3_xg_1.keras",
        custom_objects={'RandomColorDistortion': RandomColorDistortion}
    )
    model_list.append(final_test_sub_model4)
    final_test_sub_model1_2 = keras.models.load_model(
        "./models/try_cnn_xgboost_model_2_1_2.keras",
        custom_objects={'RandomColorDistortion': RandomColorDistortion}
    )
    model_list.append(final_test_sub_model1_2)
    final_test_sub_model2_2 = keras.models.load_model(
        "./models/try_cnn_xgboost_model_2_2_2.keras",
        custom_objects={'RandomColorDistortion': RandomColorDistortion}
    )
    model_list.append(final_test_sub_model2_2)
    final_test_sub_model3_2 = keras.models.load_model(
        "./models/try_cnn_xgboost_model_2_3_2.keras",
        custom_objects={'RandomColorDistortion': RandomColorDistortion}
    )
    model_list.append(final_test_sub_model3_2)
    final_test_sub_model1_3 = keras.models.load_model(
        "./models/try_cnn_xgboost_model_2_1_3.keras",
        custom_objects={'RandomColorDistortion': RandomColorDistortion}
    )
    model_list.append(final_test_sub_model1_3)
    final_test_sub_model2_3 = keras.models.load_model(
        "./models/try_cnn_xgboost_model_2_2_3.keras",
        custom_objects={'RandomColorDistortion': RandomColorDistortion}
    )
    model_list.append(final_test_sub_model2_3)
    final_test_sub_model3_3 = keras.models.load_model(
        "./models/try_cnn_xgboost_model_2_3_3.keras",
        custom_objects={'RandomColorDistortion': RandomColorDistortion}
    )
    model_list.append(final_test_sub_model3_3)
    final_test_sub_model1_4 = keras.models.load_model(
        "./models/try_cnn_xgboost_model_2_1_4.keras",
        custom_objects={'RandomColorDistortion': RandomColorDistortion}
    )
    model_list.append(final_test_sub_model1_4)
    final_test_sub_model2_4 = keras.models.load_model(
        "./models/try_cnn_xgboost_model_2_2_4.keras",
        custom_objects={'RandomColorDistortion': RandomColorDistortion}
    )
    model_list.append(final_test_sub_model2_4)
    final_test_sub_model3_4 = keras.models.load_model(
        "./models/try_cnn_xgboost_model_2_3_4.keras",
        custom_objects={'RandomColorDistortion': RandomColorDistortion}
    )
    model_list.append(final_test_sub_model3_4)

    final_xgbrmodel = joblib.load("./models/2xgbr_model_xg_all")
    model_list.append(final_xgbrmodel)

    return model_list

model_list = load_model()

class Config(object):
    xg_weight = 0.625793
    main_k_weight = 0.15442506
    main_weight = 0.16338391
par2 = Config()

st.subheader('Some examples:')
st.write('To see the effect, you can drag and drop the examples to the uploader')
example1, example2, example3, example4 = st.columns(4)
example1.image(io.imread('E_j31.521noise.png'), caption='(E_J / E_C) = 31.521')
example2.image(io.imread('ex011.png'), caption='(E_J / E_C) = 66.67')
example3.image(io.imread('ex046.png'), caption='(E_J / E_C) = 94.44')
example4.image(io.imread('E_j119.964noise.png'), caption='(E_J / E_C) = 119.964')


# 上傳圖檔
uploaded_file = st.file_uploader("Upload the image(.png)", type="png")
if uploaded_file is not None:
    if method==1:
        preds = CNN_XGBRegression_model_all(
                                            CNN_model1=model_list[0],
                                            CNN_model2=model_list[1],
                                            CNN_model3=model_list[2],
                                            CNN_model4=model_list[3],
                                            CNN_model5=model_list[4],
                                            CNN_model6=model_list[5],
                                            CNN_model7=model_list[6],
                                            CNN_model8=model_list[7],
                                            CNN_model9=model_list[8],
                                            CNN_model10=model_list[9],
                                            CNN_model11=model_list[10],
                                            CNN_model12=model_list[11],
                                            CNN_model13=model_list[12],
                                            CNN_model14=model_list[13],
                                            XGBR_model=model_list[14],
                                            input_img_path=uploaded_file)
    if method==2:
        preds_1 = CNN_XGBRegression_model_all(
            CNN_model1=model_list[0],
            CNN_model2=model_list[1],
            CNN_model3=model_list[2],
            CNN_model4=model_list[3],
            CNN_model5=model_list[4],
            CNN_model6=model_list[5],
            CNN_model7=model_list[6],
            CNN_model8=model_list[7],
            CNN_model9=model_list[8],
            CNN_model10=model_list[9],
            CNN_model11=model_list[10],
            CNN_model12=model_list[11],
            CNN_model13=model_list[12],
            CNN_model14=model_list[13],
            XGBR_model=model_list[14],
            input_img_path=uploaded_file)
        inputs_img = get_img_array(uploaded_file, target_size=(224, 224))
        preds_2 = model_list[0].predict(inputs_img)[0][0]
        preds_3 = model_list[4].predict(inputs_img)[0][0]
        preds = preds_1 * par2.xg_weight + preds_2 * par2.main_k_weight + preds_3 * par2.main_weight
    st.write(f'## Prediction: {preds}')
    # 顯示上傳圖檔
    image1 = io.imread(uploaded_file)
    st.image(image1)

st.write("""
***
""")
st.image(io.imread('partner_logo_nthu.gif'))












