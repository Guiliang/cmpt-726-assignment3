'''Code for fine-tuning Inception V3 for a new task.

Start with Inception V3 network, not including last fully connected layers.

Train a simple fully connected layer on top of these.


'''
import random
import html_generates as hg

import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout
import inception_v3 as inception

N_CLASSES = 3
IMSIZE = (299, 299)

# TO DO:: Replace these with paths to the downloaded data.
# Training directory
train_dir = '/Users/liu/Desktop/CMPT 726 Machine Learning material/sport3/train'
# Testing directory
test_dir = '/Users/liu/Desktop/CMPT 726 Machine Learning material/sport3/validation'

# Start with an Inception V3 model, not including the final softmax layer.
base_model = inception.InceptionV3(weights='imagenet')
print 'Loaded Inception model'

# Turn off training on base model layers
# for layer in base_model.layers:
#     layer.trainable = False

# Add on new fully connected layers for the output classes.
x = Dense(32, activation='relu')(base_model.get_layer('flatten').output)
x = Dropout(0.5)(x)
predictions = Dense(N_CLASSES, activation='softmax', name='predictions')(x)

model = Model(input=base_model.input, output=predictions)

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Show some debug output
print (model.summary())

print 'Trainable weights'
print model.trainable_weights

# Data generators for feeding training/testing images to the model.
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_dir,  # this is the target directory
    target_size=IMSIZE,  # all images will be resized to 299x299 Inception V3 input
    batch_size=32,
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,  # this is the target directory
    target_size=IMSIZE,  # all images will be resized to 299x299 Inception V3 input
    batch_size=32,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    samples_per_epoch=32,
    # samples_per_epoch=200,

    nb_epoch=5,
    validation_data=test_generator,
    verbose=2,
    nb_val_samples=80)
model.save_weights('sport3_pretrain.h5')  # always save your weights after training or during training

# store all the result
test_result_list_all = []

# test sccoer
for sccoer_num in range(0, 10):
    test_sccoer_num = random.randrange(100, 1000)
    img_path = '/Users/liu/Desktop/CMPT 726 Machine Learning material/sport3/validation/soccer/img_' + str(
        test_sccoer_num) + '.jpg'
    img = image.load_img(img_path, target_size=IMSIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = inception.preprocess_input(x)
    preds = model.predict(x)
    print('Predicted sccoer:', preds)
    preds_list = preds[0].tolist()
    preds_list.append(img_path)
    test_result_list_all.append(preds_list)

# test basketball
for basketball_num in range(0, 10):
    test_basketball_num = random.randrange(2000, 3000)
    img_path = '/Users/liu/Desktop/CMPT 726 Machine Learning material/sport3/validation/basketball/img_' + str(
        test_basketball_num) + '.jpg'
    img = image.load_img(img_path, target_size=IMSIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = inception.preprocess_input(x)
    preds = model.predict(x)
    print('Predicted basketball:', preds)
    preds_list = preds[0].tolist()
    preds_list.append(img_path)
    test_result_list_all.append(preds_list)

# test hockey
for hockey_num in range(0, 10):
    test_hockey_num = random.randrange(2000, 3000)
    img_path = '/Users/liu/Desktop/CMPT 726 Machine Learning material/sport3/validation/hockey/img_' + str(
        test_hockey_num) + '.jpg'
    img = image.load_img(img_path, target_size=IMSIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = inception.preprocess_input(x)
    preds = model.predict(x)
    print('Predicted hockey:', preds)
    preds_list = preds[0].tolist()
    preds_list.append(img_path)
    test_result_list_all.append(preds_list)

print(test_result_list_all)
html = hg.generate_html(test_result_list_all)
hg.write_lines(html, "predict_table.html")
# preds_batch = model.predict_on_batch(x)
# print ('Predicted batch', preds_batch)
