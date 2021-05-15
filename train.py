from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Conv2DTranspose
from keras.models import Input
from keras import Model
from keras.preprocessing.image import load_img
from keras.layers import concatenate
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(description='train unet.')
parser.add_argument("--config", help="Config file for UNET", default='./config.yaml')
args = parser.parse_args()
# print(args.config)

def directory_to_generator(img_path , mask_path , image_size):
	data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
	image_datagen = ImageDataGenerator(**data_gen_args)
	mask_datagen = ImageDataGenerator(**data_gen_args)
	img_generator = image_datagen.flow_from_directory(
        img_path,
        target_size=(image_size, image_size),
        class_mode=None,seed = 42)
	mask_generator = mask_datagen.flow_from_directory(
        mask_path,
        target_size=(image_size, image_size),
        seed = 42,
        class_mode=None)
	image_datagen.fit(img_generator[0])
	mask_datagen.fit(mask_generator[0])
	train_generator = zip(img_generator, mask_generator)
	return train_generator

def build_model(input_layer, start_neurons):
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="lrelu", padding="same")(input_layer)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="lrelu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="lrelu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="lrelu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="lrelu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="lrelu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="lrelu", padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="lrelu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation="lrelu", padding="same")(pool4)
    convm = Conv2D(start_neurons * 16, (3, 3), activation="lrelu", padding="same")(convm)
    
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="lrelu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="lrelu", padding="same")(uconv4)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="lrelu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="lrelu", padding="same")(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="lrelu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="lrelu", padding="same")(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="lrelu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="lrelu", padding="same")(uconv1)
    
    output_layer = Conv2D(3, (1,1), padding="same", activation="sigmoid")(uconv1)
    
    return output_layer

def dice_loss(onehots_true, logits):
    probabilities = tf.nn.softmax(logits)
	   
    numerator = tf.reduce_sum(onehots_true * probabilities, axis=0)
	   
    denominator = tf.reduce_sum(onehots_true + probabilities, axis=0)
	   
    loss = 1.0 - 2.0 * (numerator + 1) / (denominator + 1)
    return loss

def main()
	with open(args.config, 'r') as f:
        yam = yaml.load(f)
    img_path = yam['img_path']
    mask_path = yam['mask_path']
    epochs = yam['epochs']
    image_size = yam['image_size']
    start_neurons = yam['start_neurons']
    batch_size = yam['batch_size']
    get_custom_objects().update({'lrelu': Activation(tf.keras.layers.LeakyReLU(alpha=0.3))})
	train_generator = directory_to_generator(img_path , mask_path , image_size)
	steps_per_epoch = int( np.ceil(train_generator.shape[0] / batch_size) )
	input_layer = Input((image_size, image_size, 3))
	output_layer = build_model(input_layer, start_neurons)
	model = Model(input_layer, output_layer)
	model.compile(loss = dice_loss, optimizer='adam', metrics=["accuracy"])
	model.fit(train_generator , epochs = epochs , steps_per_epoch = steps_per_epoch , batch_size = batch_size)

if __name__ == "__main__":
    main()


