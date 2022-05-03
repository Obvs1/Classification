from ..SqueezeNet.squeezenet import SqueezeNet
from ..ResNet.resnet50 import ResNet50
from ..InceptionV3.inceptionv3 import InceptionV3
from ..DenseNet.densenet import DenseNetImageNet121
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.layers import Flatten, Dense, Input, Conv2D, GlobalAvgPool2D, Activation
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import load_model, save_model

from tensorflow.python.keras import backend as K
from PIL import Image
from tensorflow.python.keras.callbacks import TensorBoard
import os
from tensorflow.python.keras.callbacks import ModelCheckpoint
from io import open
import json
import numpy as np
import warnings
import tensorflow as tf
from tensorflow import keras

import numpy as np
from ..SqueezeNet.squeezenet import SqueezeNet
from ..ResNet.resnet50 import ResNet50
from ..InceptionV3.inceptionv3 import InceptionV3
from ..DenseNet.densenet import DenseNetImageNet121
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.layers import Flatten, Dense, Input, Conv2D, GlobalAvgPool2D, Activation
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import load_model, save_model

from tensorflow.python.keras import backend as K
from PIL import Image
from tensorflow.python.keras.callbacks import TensorBoard
import os
from tensorflow.python.keras.callbacks import ModelCheckpoint
from io import open
import json
import numpy as np
import warnings
import tensorflow as tf
from tensorflow import keras





class ModelTraining:
    """
        This is the Model training class, that allows you to define a deep learning network
        from the 4 available networks types supported by ImageAI which are SqueezeNet, ResNet50,
        InceptionV3 and DenseNet121. Once you instantiate this class, you must call:

        *
    """

    def __init__(self):
        self.__modelType = ""
        self.__use_pretrained_model = False
        self.__data_dir = ""
        self.__train_dir = ""
        self.__test_dir = ""
        self.__num_epochs = 10
        self.__trained_model_dir = ""
        self.__model_class_dir = ""
        self.__initial_learning_rate = 1e-3
        self.__model_collection = []




    def setModelTypeAsSqueezeNet(self):
        """
        'setModelTypeAsSqueezeNet()' is used to set the model type to the SqueezeNet model
        for the training instance object .
        :return:
        """
        self.__modelType = "squeezenet"

    def setModelTypeAsResNet(self):
        """
         'setModelTypeAsResNet()' is used to set the model type to the ResNet model
                for the training instance object .
        :return:
        """
        self.__modelType = "resnet"

    def setModelTypeAsDenseNet(self):
        """
         'setModelTypeAsDenseNet()' is used to set the model type to the DenseNet model
                for the training instance object .
        :return:
        """
        self.__modelType = "densenet"

    def setModelTypeAsInceptionV3(self):
        """
         'setModelTypeAsInceptionV3()' is used to set the model type to the InceptionV3 model
                for the training instance object .
        :return:
        """
        self.__modelType = "inceptionv3"

    def setDataDirectory(self, data_directory="", train_subdirectory="train", test_subdirectory="test",
                         models_subdirectory="model", json_subdirectory="json"):

        self.__data_dir = data_directory
        self.__train_dir = os.path.join(self.__data_dir, train_subdirectory)
        self.__test_dir = os.path.join(self.__data_dir, test_subdirectory)
        self.__trained_model_dir = os.path.join(str(models_subdirectory))      
        self.__model_class_dir = os.path.join(str(json_subdirectory))   

    def lr_schedule(self, epoch):

        # Learning Rate Schedule


        lr = self.__initial_learning_rate
        total_epochs = self.__num_epochs

        check_1 = int(total_epochs * 0.9)
        check_2 = int(total_epochs * 0.8)
        check_3 = int(total_epochs * 0.6)
        check_4 = int(total_epochs * 0.4)

        if epoch > check_1:
            lr *= 1e-4
        elif epoch > check_2:
            lr *= 1e-3
        elif epoch > check_3:
            lr *= 1e-2
        elif epoch > check_4:
            lr *= 1e-1


        return lr




    def trainModel(self, num_objects,num_experiments=200, enhance_data=False, batch_size = 32, initial_learning_rate=1e-3, show_network_summary=False, training_image_size = 224, continue_from_model=None, transfer_from_model=None, transfer_with_full_training=True, initial_num_objects = None, save_full_model = False):


        self.__num_epochs = num_experiments
        self.__initial_learning_rate = initial_learning_rate
        lr_scheduler = LearningRateScheduler(self.lr_schedule)


        num_classes = num_objects

        if(training_image_size < 100):
            warnings.warn("The specified training_image_size {} is less than 100. Hence the training_image_size will default to 100.".format(training_image_size))
            training_image_size = 100



        image_input = Input(shape=(training_image_size, training_image_size, 3))
        if (self.__modelType == "squeezenet"):
            if (continue_from_model != None):
                model = SqueezeNet(weights="continued", num_classes=num_classes, model_input=image_input, model_path=continue_from_model)
                if (show_network_summary == True):
                    print("Resuming training with weights loaded from a previous model")

            elif (transfer_from_model != None):
                model = SqueezeNet(weights="transfer", num_classes=num_classes, model_input=image_input,
                                   model_path=transfer_from_model, initial_num_classes=initial_num_objects,transfer_with_full_training=transfer_with_full_training)
                if (show_network_summary == True):
                    print("Training using weights from a pre-trained model")
            else:
                model = SqueezeNet(weights="custom", num_classes=num_classes, model_input=image_input)
        elif (self.__modelType == "resnet"):
            if(continue_from_model != None):
                model = ResNet50(weights="continued", num_classes=num_classes, model_input=image_input, model_path=continue_from_model)
                if (show_network_summary == True):
                    print("Resuming training with weights loaded from a previous model")
            elif(transfer_from_model != None):
                model = ResNet50(weights="transfer", num_classes=num_classes, model_input=image_input, model_path=transfer_from_model, initial_num_classes=initial_num_objects, transfer_with_full_training=transfer_with_full_training)
                if (show_network_summary == True):
                    print("Training using weights from a pre-trained model")
            else:
                model = ResNet50(weights="custom", num_classes=num_classes, model_input=image_input)

        elif (self.__modelType == "inceptionv3"):
            if (continue_from_model != None):
                model = InceptionV3(weights="continued", classes=num_classes, model_input=image_input, model_path=continue_from_model)
                if (show_network_summary == True):
                    print("Resuming training with weights loaded from a previous model")
            elif (transfer_from_model != None):
                model = InceptionV3(weights="transfer", classes=num_classes, model_input=image_input,
                                    model_path=transfer_from_model, initial_classes=initial_num_objects,
                                 transfer_with_full_training=transfer_with_full_training)
                if (show_network_summary == True):
                    print("Training using weights from a pre-trained model")
            else:
                model = InceptionV3(weights="custom", classes=num_classes, model_input=image_input)

        elif (self.__modelType == "densenet"):
            if (continue_from_model != None):
                model = DenseNetImageNet121(weights="continued", classes=num_classes, model_input=image_input, model_path=continue_from_model)
                if (show_network_summary == True):
                    print("Resuming training with weights loaded from a previous model")
            elif (transfer_from_model != None):
                model = DenseNetImageNet121(weights="transfer", classes=num_classes, model_input=image_input,
                                    model_path=transfer_from_model, initial_num_classes=initial_num_objects,
                                 transfer_with_full_training=transfer_with_full_training)
                if (show_network_summary == True):
                    print("Training using weights from a pre-trained model")
            else:
                model = DenseNetImageNet121(weights="custom", classes=num_classes, model_input=image_input)


        optimizer = Adam(lr=self.__initial_learning_rate, decay=1e-4)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        print(model.output.op.name)
        print(model.input.op.name)
       
        saver = tf.train.Saver()
        saver.save(K.get_session(), './tmp/keras_model.ckpt')
        

        

        if (show_network_summary == True):
            model.summary()

        model_name = 'model_ex-{epoch:03d}_acc-{val_acc:03f}.h5'
        top_layers_checkpoint_path = 'cp.top.best.hdf5'
        if not os.path.isdir(self.__trained_model_dir):
            os.makedirs(self.__trained_model_dir)

        if not os.path.isdir(self.__model_class_dir):
            os.makedirs(self.__model_class_dir)

        model_path = os.path.join(self.__trained_model_dir, model_name)

        save_weights_condition = True

        if(save_full_model == True ):
            save_weights_condition = False
        elif(save_full_model == False):
            save_weights_condition = True

        checkpoint = ModelCheckpoint(filepath=model_path,
                                     monitor='val_acc',
                                     verbose=1,
                                     save_weights_only=save_weights_condition,
                                     save_best_only=True,
                                     period=1)

        if (enhance_data == True):
            print("Using Enhanced Data Generation")

        height_shift = 0
        width_shift = 0
        if (enhance_data == True):
            height_shift = 0.1
            width_shift = 0.1

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=enhance_data, height_shift_range=height_shift, width_shift_range=width_shift)

        test_datagen = ImageDataGenerator(
            rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(self.__train_dir, target_size=(training_image_size, training_image_size),
                                                            batch_size=batch_size,
                                                            class_mode="categorical")
        test_generator = test_datagen.flow_from_directory(self.__test_dir, target_size=(training_image_size, training_image_size),
                                                          batch_size=batch_size,
                                                          class_mode="categorical")

        class_indices = train_generator.class_indices
        class_json = {}
        for eachClass in class_indices:
            class_json[str(class_indices[eachClass])] = eachClass

        with open(os.path.join(self.__model_class_dir, "model_class.json"), "w+") as json_file:
            json.dump(class_json, json_file, indent=4, separators=(",", " : "),
                      ensure_ascii=True)
            json_file.close()
        print("JSON Mapping for the model classes saved to ", os.path.join(self.__model_class_dir, "model_class.json"))

        num_train = len(train_generator.filenames)
        num_test = len(test_generator.filenames)
        print("Number of experiments (Epochs) : ", self.__num_epochs)
        mc_top = ModelCheckpoint(top_layers_checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='max', period=1)
            
        

        tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

        model.fit_generator(train_generator, steps_per_epoch=int(num_train / batch_size), epochs=self.__num_epochs,
                            validation_data=test_generator,
                            validation_steps=int(num_test / batch_size), callbacks=[checkpoint,mc_top, tb,lr_scheduler])
        





class CustomImagePrediction:


    def __init__(self):
        self.__modelType = ""
        self.modelPath = ""
        self.jsonPath = ""
        self.numObjects = 10
        self.__modelLoaded = False
        self.__model_collection = []
        self.__input_image_size = 224

    def setModelPath(self, model_path):

        self.modelPath = model_path

    def setJsonPath(self, model_json):

        self.jsonPath = model_json

    def setModelTypeAsSqueezeNet(self):

        self.__modelType = "squeezenet"

    def setModelTypeAsResNet(self):

        self.__modelType = "resnet"

    def setModelTypeAsDenseNet(self):
        self.__modelType = "densenet"

    def setModelTypeAsInceptionV3(self):

        self.__modelType = "inceptionv3"

    def loadModel(self, prediction_speed="normal", num_objects=10):


        self.numObjects = num_objects

        if (prediction_speed == "normal"):
            self.__input_image_size = 224
        elif (prediction_speed == "fast"):
            self.__input_image_size = 160
        elif (prediction_speed == "faster"):
            self.__input_image_size = 120
        elif (prediction_speed == "fastest"):
            self.__input_image_size = 100

        if (self.__modelLoaded == False):

            image_input = Input(shape=(self.__input_image_size, self.__input_image_size, 3))

            if (self.__modelType == ""):
                raise ValueError("You must set a valid model type before loading the model.")


            elif (self.__modelType == "squeezenet"):
                import numpy as np
                from tensorflow.python.keras.preprocessing import image
                from ..SqueezeNet.squeezenet import SqueezeNet
                from .custom_utils import preprocess_input
                from .custom_utils import decode_predictions

                model = SqueezeNet(model_path=self.modelPath, weights="trained", model_input=image_input,
                                   num_classes=self.numObjects)
                self.__model_collection.append(model)
                self.__modelLoaded = True
                try:
                    None
                except:
                    raise ("You have specified an incorrect path to the SqueezeNet model file.")
            elif (self.__modelType == "resnet"):
                import numpy as np
                from tensorflow.python.keras.preprocessing import image
                from ..ResNet.resnet50 import ResNet50
                from .custom_utils import preprocess_input
                from .custom_utils import decode_predictions
                try:
                    model = ResNet50(model_path=self.modelPath, weights="trained", model_input=image_input, num_classes=self.numObjects)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ValueError("You have specified an incorrect path to the ResNet model file.")

            elif (self.__modelType == "densenet"):
                from tensorflow.python.keras.preprocessing import image
                from ..DenseNet.densenet import DenseNetImageNet121
                from .custom_utils import decode_predictions, preprocess_input
                import numpy as np
                try:
                    model = DenseNetImageNet121(model_path=self.modelPath, weights="trained", model_input=image_input, classes=self.numObjects)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ValueError("You have specified an incorrect path to the DenseNet model file.")

            elif (self.__modelType == "inceptionv3"):
                import numpy as np
                from tensorflow.python.keras.preprocessing import image

                from imageai.Prediction.InceptionV3.inceptionv3 import InceptionV3
                from .custom_utils import decode_predictions, preprocess_input



                try:
                    model = InceptionV3(include_top=True, weights="trained", model_path=self.modelPath,
                                        model_input=image_input, classes=self.numObjects)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ValueError("You have specified an incorrect path to the InceptionV3 model file.")

    def loadFullModel(self, prediction_speed="normal", num_objects=10):


        self.numObjects = num_objects

        if (prediction_speed == "normal"):
            self.__input_image_size = 224
        elif (prediction_speed == "fast"):
            self.__input_image_size = 160
        elif (prediction_speed == "faster"):
            self.__input_image_size = 120
        elif (prediction_speed == "fastest"):
            self.__input_image_size = 100

        if (self.__modelLoaded == False):

            image_input = Input(shape=(self.__input_image_size, self.__input_image_size, 3))


            model = load_model(filepath=self.modelPath)
            self.__model_collection.append(model)
            self.__modelLoaded = True
            self.__modelType = "full"

    def save_model_to_tensorflow(self, new_model_folder, new_model_name=""):



        if(self.__modelLoaded == True):
            out_prefix = "output_"
            output_dir = new_model_folder
            if os.path.exists(output_dir) == False:
                os.mkdir(output_dir)
            model_name = os.path.join(output_dir, new_model_name)

            keras_model = self.__model_collection[0]


            out_nodes = []

            for i in range(len(keras_model.outputs)):
                out_nodes.append(out_prefix + str(i + 1))
                tf.identity(keras_model.output[i], out_prefix + str(i + 1))

            sess = K.get_session()

            from tensorflow.python.framework import graph_util, graph_io

            init_graph = sess.graph.as_graph_def()

            main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)

            graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)
            print("Tensorflow Model Saved")

    def save_model_for_deepstack(self, new_model_folder, new_model_name=""):


        if(self.__modelLoaded == True):
            print(self.jsonPath)
            with open(self.jsonPath) as inputFile:
                model_json = json.load(inputFile)

                deepstack_json = {"sys-version": "1.0", "framework":"KERAS","mean":0.5,"std":255}
                deepstack_json["width"] = self.__input_image_size
                deepstack_json["height"] = self.__input_image_size

                deepstack_classes_map = {}


                for eachClass in model_json:
                    deepstack_classes_map[eachClass] = model_json[eachClass]

                deepstack_json["map"] = deepstack_classes_map

                output_dir = new_model_folder
                if os.path.exists(output_dir) == False:
                    os.mkdir(output_dir)

                with open(os.path.join(output_dir,"config.json"), "w+") as json_file:
                    json.dump(deepstack_json, json_file, indent=4, separators=(",", " : "),
                              ensure_ascii=True)
                    json_file.close()
                print("JSON Config file saved for DeepStack format in ",
                      os.path.join(output_dir, "config.json"))

                keras_model = self.__model_collection[0]
                save_model(keras_model, os.path.join(new_model_folder, new_model_name))
                print("Model saved for DeepStack format in",
                      os.path.join(os.path.join(new_model_folder, new_model_name)))







    def predictImage(self, image_input, result_count=1, input_type="file"):

        prediction_results = []
        prediction_probabilities = []
        if (self.__modelLoaded == False):
            raise ValueError("You must call the loadModel() function before making predictions.")

        else:

            if (self.__modelType == "squeezenet"):

                from .custom_utils import preprocess_input
                from .custom_utils import decode_predictions
                if (input_type == "file"):
                    try:
                        image_to_predict = image.load_img(image_input, target_size=(
                        self.__input_image_size, self.__input_image_size))
                        image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
                        image_to_predict = np.expand_dims(image_to_predict, axis=0)

                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have set a path to an invalid image file.")
                elif (input_type == "array"):
                    try:
                        image_input = Image.fromarray(np.uint8(image_input))
                        image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                        image_input = np.expand_dims(image_input, axis=0)
                        image_to_predict = image_input.copy()
                        image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have parsed in a wrong numpy array for the image")
                elif (input_type == "stream"):
                    try:
                        image_input = Image.open(image_input)
                        image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                        image_input = np.expand_dims(image_input, axis=0)
                        image_to_predict = image_input.copy()
                        image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have parsed in a wrong stream for the image")

                model = self.__model_collection[0]

                prediction = model.predict(image_to_predict, steps=1)

                try:
                    predictiondata = decode_predictions(prediction, top=int(result_count), model_json=self.jsonPath)

                    for result in predictiondata:
                        prediction_results.append(str(result[0]))
                        prediction_probabilities.append(str(result[1] * 100))
                except:
                    raise ValueError("An error occured! Try again.")

                return prediction_results, prediction_probabilities
            elif (self.__modelType == "resnet"):

                model = self.__model_collection[0]

                from .custom_utils import preprocess_input
                from .custom_utils import decode_predictions
                if (input_type == "file"):
                    try:
                        image_to_predict = image.load_img(image_input, target_size=(
                        self.__input_image_size, self.__input_image_size))
                        image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
                        image_to_predict = np.expand_dims(image_to_predict, axis=0)

                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have set a path to an invalid image file.")
                elif (input_type == "array"):
                    try:
                        image_input = Image.fromarray(np.uint8(image_input))
                        image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                        image_input = np.expand_dims(image_input, axis=0)
                        image_to_predict = image_input.copy()
                        image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have parsed in a wrong numpy array for the image")
                elif (input_type == "stream"):
                    try:
                        image_input = Image.open(image_input)
                        image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                        image_input = np.expand_dims(image_input, axis=0)
                        image_to_predict = image_input.copy()
                        image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have parsed in a wrong stream for the image")

                prediction = model.predict(x=image_to_predict, steps=1)




                try:

                    predictiondata = decode_predictions(prediction, top=int(result_count), model_json=self.jsonPath)

                    for result in predictiondata:
                        prediction_results.append(str(result[0]))
                        prediction_probabilities.append(str(result[1] * 100))


                except:
                    raise ValueError("An error occured! Try again.")

                return prediction_results, prediction_probabilities
            elif (self.__modelType == "densenet"):

                model = self.__model_collection[0]

                from .custom_utils import preprocess_input
                from .custom_utils import decode_predictions
                from ..DenseNet.densenet import DenseNetImageNet121
                if (input_type == "file"):
                    try:
                        image_to_predict = image.load_img(image_input, target_size=(
                        self.__input_image_size, self.__input_image_size))
                        image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
                        image_to_predict = np.expand_dims(image_to_predict, axis=0)

                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have set a path to an invalid image file.")
                elif (input_type == "array"):
                    try:
                        image_input = Image.fromarray(np.uint8(image_input))
                        image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                        image_input = np.expand_dims(image_input, axis=0)
                        image_to_predict = image_input.copy()
                        image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have parsed in a wrong numpy array for the image")
                elif (input_type == "stream"):
                    try:
                        image_input = Image.open(image_input)
                        image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                        image_input = np.expand_dims(image_input, axis=0)
                        image_to_predict = image_input.copy()
                        image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have parsed in a wrong stream for the image")

                prediction = model.predict(x=image_to_predict, steps=1)

                try:
                    predictiondata = decode_predictions(prediction, top=int(result_count), model_json=self.jsonPath)

                    for result in predictiondata:
                        prediction_results.append(str(result[0]))
                        prediction_probabilities.append(str(result[1] * 100))
                except:
                    raise ValueError("An error occured! Try again.")

                return prediction_results, prediction_probabilities
            elif (self.__modelType == "inceptionv3"):

                model = self.__model_collection[0]

                from imageai.Prediction.InceptionV3.inceptionv3 import InceptionV3
                from .custom_utils import decode_predictions, preprocess_input

                if (input_type == "file"):
                    try:
                        image_to_predict = image.load_img(image_input, target_size=(
                        self.__input_image_size, self.__input_image_size))
                        image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
                        image_to_predict = np.expand_dims(image_to_predict, axis=0)

                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have set a path to an invalid image file.")
                elif (input_type == "array"):
                    try:
                        image_input = Image.fromarray(np.uint8(image_input))
                        image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                        image_input = np.expand_dims(image_input, axis=0)
                        image_to_predict = image_input.copy()
                        image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have parsed in a wrong numpy array for the image")
                elif (input_type == "stream"):
                    try:
                        image_input = Image.open(image_input)
                        image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                        image_input = np.expand_dims(image_input, axis=0)
                        image_to_predict = image_input.copy()
                        image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have parsed in a wrong stream for the image")

                prediction = model.predict(x=image_to_predict, steps=1)

                try:
                    predictiondata = decode_predictions(prediction, top=int(result_count), model_json=self.jsonPath)

                    for result in predictiondata:
                        prediction_results.append(str(result[0]))
                        prediction_probabilities.append(str(result[1] * 100))
                except:
                    raise ValueError("An error occured! Try again.")

                return prediction_results, prediction_probabilities

            elif (self.__modelType == "full"):

                model = self.__model_collection[0]

                from imageai.Prediction.InceptionV3.inceptionv3 import InceptionV3
                from .custom_utils import decode_predictions, preprocess_input

                if (input_type == "file"):
                    try:
                        image_to_predict = image.load_img(image_input, target_size=(
                        self.__input_image_size, self.__input_image_size))
                        image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
                        image_to_predict = np.expand_dims(image_to_predict, axis=0)

                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have set a path to an invalid image file.")
                elif (input_type == "array"):
                    try:
                        image_input = Image.fromarray(np.uint8(image_input))
                        image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                        image_input = np.expand_dims(image_input, axis=0)
                        image_to_predict = image_input.copy()
                        image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have parsed in a wrong numpy array for the image")
                elif (input_type == "stream"):
                    try:
                        image_input = Image.open(image_input)
                        image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                        image_input = np.expand_dims(image_input, axis=0)
                        image_to_predict = image_input.copy()
                        image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have parsed in a wrong stream for the image")

                prediction = model.predict(x=image_to_predict, steps=1)

                try:
                    predictiondata = decode_predictions(prediction, top=int(result_count), model_json=self.jsonPath)

                    for result in predictiondata:
                        prediction_results.append(str(result[0]))
                        prediction_probabilities.append(str(result[1] * 100))
                except:
                    raise ValueError("An error occured! Try again.")

                return prediction_results, prediction_probabilities




    def predictMultipleImages(self, sent_images_array, result_count_per_image=1, input_type="file"):


        output_array = []

        for image_input in sent_images_array:

            prediction_results = []
            prediction_probabilities = []
            if (self.__modelLoaded == False):
                raise ValueError("You must call the loadModel() function before making predictions.")

            else:
                if (self.__modelType == "squeezenet"):

                    from .custom_utils import preprocess_input
                    from .custom_utils import decode_predictions
                    if (input_type == "file"):
                        try:
                            image_to_predict = image.load_img(image_input, target_size=(
                                self.__input_image_size, self.__input_image_size))
                            image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
                            image_to_predict = np.expand_dims(image_to_predict, axis=0)

                            image_to_predict = preprocess_input(image_to_predict)
                        except:
                            raise ValueError("You have set a path to an invalid image file.")
                    elif (input_type == "array"):
                        try:
                            image_input = Image.fromarray(np.uint8(image_input))
                            image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                            image_input = np.expand_dims(image_input, axis=0)
                            image_to_predict = image_input.copy()
                            image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                            image_to_predict = preprocess_input(image_to_predict)
                        except:
                            raise ValueError("You have parsed in a wrong numpy array for the image")
                    elif (input_type == "stream"):
                        try:
                            image_input = Image.open(image_input)
                            image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                            image_input = np.expand_dims(image_input, axis=0)
                            image_to_predict = image_input.copy()
                            image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                            image_to_predict = preprocess_input(image_to_predict)
                        except:
                            raise ValueError("You have parsed in a wrong stream for the image")

                    model = self.__model_collection[0]

                    prediction = model.predict(image_to_predict, steps=1)

                    try:
                        predictiondata = decode_predictions(prediction, top=int(result_count_per_image), model_json=self.jsonPath)

                        for result in predictiondata:
                            prediction_results.append(str(result[0]))
                            prediction_probabilities.append(str(result[1] * 100))
                    except:
                        raise ValueError("An error occured! Try again.")

                    each_image_details = {}
                    each_image_details["predictions"] = prediction_results
                    each_image_details["percentage_probabilities"] = prediction_probabilities
                    output_array.append(each_image_details)

                elif (self.__modelType == "resnet"):

                    model = self.__model_collection[0]

                    from .custom_utils import preprocess_input
                    from .custom_utils import decode_predictions
                    if (input_type == "file"):
                        try:
                            image_to_predict = image.load_img(image_input, target_size=(
                                self.__input_image_size, self.__input_image_size))
                            image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
                            image_to_predict = np.expand_dims(image_to_predict, axis=0)

                            image_to_predict = preprocess_input(image_to_predict)
                        except:
                            raise ValueError("You have set a path to an invalid image file.")
                    elif (input_type == "array"):
                        try:
                            image_input = Image.fromarray(np.uint8(image_input))
                            image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                            image_input = np.expand_dims(image_input, axis=0)
                            image_to_predict = image_input.copy()
                            image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                            image_to_predict = preprocess_input(image_to_predict)
                        except:
                            raise ValueError("You have parsed in a wrong numpy array for the image")
                    elif (input_type == "stream"):
                        try:
                            image_input = Image.open(image_input)
                            image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                            image_input = np.expand_dims(image_input, axis=0)
                            image_to_predict = image_input.copy()
                            image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                            image_to_predict = preprocess_input(image_to_predict)
                        except:
                            raise ValueError("You have parsed in a wrong stream for the image")

                    prediction = model.predict(x=image_to_predict, steps=1)

                    try:

                        predictiondata = decode_predictions(prediction, top=int(result_count_per_image), model_json=self.jsonPath)

                        for result in predictiondata:
                            prediction_results.append(str(result[0]))
                            prediction_probabilities.append(str(result[1] * 100))


                    except:
                        raise ValueError("An error occured! Try again.")

                    each_image_details = {}
                    each_image_details["predictions"] = prediction_results
                    each_image_details["percentage_probabilities"] = prediction_probabilities
                    output_array.append(each_image_details)


                elif (self.__modelType == "densenet"):

                    model = self.__model_collection[0]

                    from .custom_utils import preprocess_input
                    from .custom_utils import decode_predictions
                    from ..DenseNet.densenet import DenseNetImageNet121
                    if (input_type == "file"):
                        try:
                            image_to_predict = image.load_img(image_input, target_size=(
                                self.__input_image_size, self.__input_image_size))
                            image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
                            image_to_predict = np.expand_dims(image_to_predict, axis=0)

                            image_to_predict = preprocess_input(image_to_predict)
                        except:
                            raise ValueError("You have set a path to an invalid image file.")
                    elif (input_type == "array"):
                        try:
                            image_input = Image.fromarray(np.uint8(image_input))
                            image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                            image_input = np.expand_dims(image_input, axis=0)
                            image_to_predict = image_input.copy()
                            image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                            image_to_predict = preprocess_input(image_to_predict)
                        except:
                            raise ValueError("You have parsed in a wrong numpy array for the image")
                    elif (input_type == "stream"):
                        try:
                            image_input = Image.open(image_input)
                            image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                            image_input = np.expand_dims(image_input, axis=0)
                            image_to_predict = image_input.copy()
                            image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                            image_to_predict = preprocess_input(image_to_predict)
                        except:
                            raise ValueError("You have parsed in a wrong stream for the image")

                    prediction = model.predict(x=image_to_predict, steps=1)

                    try:
                        predictiondata = decode_predictions(prediction, top=int(result_count_per_image), model_json=self.jsonPath)

                        for result in predictiondata:
                            prediction_results.append(str(result[0]))
                            prediction_probabilities.append(str(result[1] * 100))
                    except:
                        raise ValueError("An error occured! Try again.")

                    each_image_details = {}
                    each_image_details["predictions"] = prediction_results
                    each_image_details["percentage_probabilities"] = prediction_probabilities
                    output_array.append(each_image_details)


                elif (self.__modelType == "inceptionv3"):

                    model = self.__model_collection[0]

                    from imageai.Prediction.InceptionV3.inceptionv3 import InceptionV3
                    from .custom_utils import decode_predictions, preprocess_input

                    if (input_type == "file"):
                        try:
                            image_to_predict = image.load_img(image_input, target_size=(
                                self.__input_image_size, self.__input_image_size))
                            image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
                            image_to_predict = np.expand_dims(image_to_predict, axis=0)

                            image_to_predict = preprocess_input(image_to_predict)
                        except:
                            raise ValueError("You have set a path to an invalid image file.")
                    elif (input_type == "array"):
                        try:
                            image_input = Image.fromarray(np.uint8(image_input))
                            image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                            image_input = np.expand_dims(image_input, axis=0)
                            image_to_predict = image_input.copy()
                            image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                            image_to_predict = preprocess_input(image_to_predict)
                        except:
                            raise ValueError("You have parsed in a wrong numpy array for the image")
                    elif (input_type == "stream"):
                        try:
                            image_input = Image.open(image_input)
                            image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                            image_input = np.expand_dims(image_input, axis=0)
                            image_to_predict = image_input.copy()
                            image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                            image_to_predict = preprocess_input(image_to_predict)
                        except:
                            raise ValueError("You have parsed in a wrong stream for the image")

                    prediction = model.predict(x=image_to_predict, steps=1)

                    try:
                        predictiondata = decode_predictions(prediction, top=int(result_count_per_image), model_json=self.jsonPath)

                        for result in predictiondata:
                            prediction_results.append(str(result[0]))
                            prediction_probabilities.append(str(result[1] * 100))
                    except:
                        raise ValueError("An error occured! Try again.")

                    each_image_details = {}
                    each_image_details["predictions"] = prediction_results
                    each_image_details["percentage_probabilities"] = prediction_probabilities
                    output_array.append(each_image_details)


        return output_array



