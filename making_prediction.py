from pathlib import Path
from keras.preprocessing import  image
import numpy as np
from keras.models import model_from_json
class_labels = [ "Plane",
                 "Car",
                 "Bird",
                 "Cat",
                 "Deer",
                 "Dog",
                 "Frog",
                 "Horse",
                 "Boat",
                 "Truck"]

f = Path("model_structure.json")
model_structure = f.read_text()
model = model_from_json(model_structure)
model.load_weights("model_weights.h5")

img = image.load_img("dog.jpg", target_size=(32,32))

image_to_test = image.img_to_array(img)

list = np.expand_dims(image_to_test, axis=0)

results = model.predict(list)

single_result = results[0]

most_likely_class_index = int(np.argmax(single_result))
class_liklihood = single_result[most_likely_class_index]

class_label = class_labels[most_likely_class_index]

print("this image is a {} and the likelihood is {:2f}".format(class_label,class_liklihood))

