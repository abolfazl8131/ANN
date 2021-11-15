import numpy as np
from keras.preprocessing import image
from keras.applications import vgg16

model = vgg16.VGG16()

img = image.load_img('unnamed.jpg', target_size=(224, 224))

x = image.img_to_array(img)

x = np.expand_dims(img, axis=0)

x = vgg16.preprocess_input(x)

predictions = model.predict(x)

predicted_classes = vgg16.decode_predictions(predictions, top=9)

for img,likelihood in predicted_classes[0]:
    print(f' this image is {img} and the likelihood is {likelihood}')