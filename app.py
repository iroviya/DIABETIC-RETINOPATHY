import streamlit as st
from PIL import Image
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import cv2


def model1(path):
    InceptionResNetV2 = tf.keras.applications.InceptionResNetV2(input_shape=(75, 75, 3), include_top=False)
    for layer in InceptionResNetV2.layers[:-10]:
        layer.trainable = False
    flatten = Flatten()(InceptionResNetV2.output)
    drop = Dropout(0.5)(flatten)
    dense1 = Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(seed=32),
                   kernel_regularizer='l2')(drop)
    BN1 = tf.keras.layers.BatchNormalization()(dense1)
    dense2 = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(seed=32),
                   kernel_regularizer='l2')(BN1)
    BN2 = tf.keras.layers.BatchNormalization()(dense2)
    dense3 = Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(seed=32),
                   kernel_regularizer='l2')(BN2)
    BN3 = tf.keras.layers.BatchNormalization()(dense3)
    Output_layer = Dense(units=2, activation='softmax', kernel_initializer=tf.keras.initializers.glorot_uniform(),
                         name='Output')(BN3)
    model4 = Model(inputs=InceptionResNetV2.input, outputs=Output_layer)
    model4.load_weights(path)
    return model4


def model2(path):
    DenseNet121_model = tf.keras.applications.DenseNet121(include_top=False, weights=None,
                                                          input_tensor=Input(shape=(256, 256, 3)))
    p = GlobalAveragePooling2D()(DenseNet121_model.output)
    d11 = Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(p)
    o1 = Dense(units=4, activation='softmax')(d11)
    model = Model(inputs=DenseNet121_model.input, outputs=o1)
    model.load_weights(path)
    return model


model1 = model1('final1.h5')
model2 = model2('final2.h5')
df = pd.DataFrame({'id_code': ['temp.png']})

st.title("APTOS 2019 Blindness Detection")
st.header("Detect diabetic retinopathy to stop blindness before it's too late")


def predict1(model):
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_dataframe(df, "", x_col='id_code', y_col=None,
                                                        target_size=(75, 75), class_mode=None, batch_size=64,
                                                        shuffle=False)
    y_predict = np.argmax(model.predict(train_generator), axis=1)
    return y_predict[0] ^ 1


def predict2(model):
    X = np.empty((1, 256, 256, 3))
    for index, entry in df.iterrows():
        img = cv2.imread("" + entry["id_code"])
        X[index, :] = cv2.resize(img, (256, 256))
        X[index, :] = X[index, :] / 255.0
    y_predict = np.argmax(model.predict(X), axis=1)
    return y_predict[0] + 1


choices = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

uploaded_file = st.file_uploader("Choose a image ", type="png")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    data = st.info("Predicting Disease......")
    image.save("temp.png")
    result1 = predict1(model1)
    if result1 == 0:
        data.warning(choices[result1])
    else:
        result2 = predict2(model2)
        data.warning(choices[result2])

