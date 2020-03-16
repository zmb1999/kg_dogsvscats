import os,math
import h5py
from keras import Input, Model
from keras.applications import ResNet50, InceptionV3, inception_v3, Xception, xception
from keras.layers import Lambda, GlobalAveragePooling2D
from keras_preprocessing.image import ImageDataGenerator

models = ["ResNet50","InceptionV3","Xception"]
def extfeather():

    os.chdir(r"D:\\machine_learning\\dogvscat_data\\")

    def write_gap(MODEL, size,i,lambda_func=None):
        print(models[i])
        input_tensor = Input(shape=(size[0], size[1], 3))
        x = input_tensor
        if lambda_func:
            x = Lambda(lambda_func)(x)

        base_model = MODEL(include_top=False, input_tensor=x, weights='imagenet')
        model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

        gen = ImageDataGenerator()
        train_generator = gen.flow_from_directory("train2", size, shuffle=False,
                                                  batch_size=32)
        test_generator = gen.flow_from_directory("test2", size, shuffle=False,
                                                 batch_size=32, class_mode=None)

        # print(train_generator.samples)
        train = model.predict_generator(train_generator, math.ceil(train_generator.samples*1.0/train_generator.batch_size), verbose=1)
        test = model.predict_generator(test_generator, math.ceil(test_generator.samples*1.0/test_generator.batch_size),verbose=1)
        with h5py.File("gap_%s.h5" % models[i]) as h:
            h.create_dataset("train", data=train)
            h.create_dataset("test", data=test)
            h.create_dataset("label", data=train_generator.classes)

    write_gap(ResNet50, (224, 224), 0)
    write_gap(InceptionV3, (299, 299), 1, inception_v3.preprocess_input)
    write_gap(Xception, (299, 299), 2, xception.preprocess_input)

if __name__ == '__main__':

    extfeather()
