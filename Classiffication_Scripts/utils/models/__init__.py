import numpy as np
from tensorflow.python.keras.preprocessing import image
from PIL import Image


from tensorflow.python.keras.layers import Input, Conv2D, MaxPool2D, Activation, concatenate, Dropout
from tensorflow.python.keras.layers import GlobalAvgPool2D, GlobalMaxPool2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Sequential


class ImagePrediction:


    def __init__(self):
        self.__modelType = ""
        self.modelPath = ""
        self.__modelLoaded = False
        self.__model_collection = []
        self.__input_image_size = 229


    def setModelPath(self, model_path):

        self.modelPath = model_path


    def setModelTypeAsSqueezeNet(self):

        self.__modelType = "squeezenet"

    def setModelTypeAsResNet(self):

        self.__modelType = "resnet"

    def setModelTypeAsDenseNet(self):

        self.__modelType = "densenet"

    def setModelTypeAsInceptionV3(self):

        self.__modelType = "inceptionv3"

    def loadModel(self, prediction_speed="normal"):


        if(prediction_speed=="normal"):
            self.__input_image_size = 224
        elif(prediction_speed=="fast"):
            self.__input_image_size = 160
        elif(prediction_speed=="faster"):
            self.__input_image_size = 120
        elif (prediction_speed == "fastest"):
            self.__input_image_size = 100

        if (self.__modelLoaded == False):

            image_input = Input(shape=(self.__input_image_size, self.__input_image_size, 3))

            if(self.__modelType == "" ):
                raise ValueError("You must set a valid model type before loading the model.")


            elif(self.__modelType == "squeezenet"):
                import numpy as np
                from tensorflow.python.keras.preprocessing import image
                from .SqueezeNet.squeezenet import SqueezeNet
                from .imagenet_utils import preprocess_input, decode_predictions
                try:
                    model = SqueezeNet(model_path=self.modelPath, model_input=image_input)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ("You have specified an incorrect path to the SqueezeNet model file.")
            elif(self.__modelType == "resnet"):
                import numpy as np
                from tensorflow.python.keras.preprocessing import image
                from .ResNet.resnet50 import ResNet50
                from .imagenet_utils import preprocess_input, decode_predictions
                try:
                    model = ResNet50(model_path=self.modelPath, model_input=image_input)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ValueError("You have specified an incorrect path to the ResNet model file.")

            elif (self.__modelType == "densenet"):
                from tensorflow.python.keras.preprocessing import image
                from .DenseNet.densenet import DenseNetImageNet121, preprocess_input, decode_predictions
                import numpy as np
                try:
                    model = DenseNetImageNet121(model_path=self.modelPath, model_input=image_input)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ValueError("You have specified an incorrect path to the DenseNet model file.")

            elif (self.__modelType == "inceptionv3"):
                import numpy as np
                from tensorflow.python.keras.preprocessing import image

                from imageai.Prediction.InceptionV3.inceptionv3 import InceptionV3
                from imageai.Prediction.InceptionV3.inceptionv3 import preprocess_input, decode_predictions

                try:
                    model = InceptionV3(include_top=True, weights="imagenet", model_path=self.modelPath, model_input=image_input)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ValueError("You have specified an incorrect path to the InceptionV3 model file.")

                





            
    def predictImage(self, image_input, result_count=5, input_type="file" ):

        prediction_results = []
        prediction_probabilities = []
        if (self.__modelLoaded == False):
            raise ValueError("You must call the loadModel() function before making predictions.")

        else:

            if (self.__modelType == "squeezenet"):

                from .imagenet_utils import preprocess_input, decode_predictions
                if (input_type == "file"):
                    try:
                        image_to_predict = image.load_img(image_input, target_size=(self.__input_image_size, self.__input_image_size))
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
                    predictiondata = decode_predictions(prediction, top=int(result_count))

                    for results in predictiondata:
                        countdown = 0
                        for result in results:
                            countdown += 1
                            prediction_results.append(str(result[1]))
                            prediction_probabilities.append(result[2] * 100)
                except:
                    raise ValueError("An error occured! Try again.")

                return prediction_results, prediction_probabilities
            elif (self.__modelType == "resnet"):

                model = self.__model_collection[0]

                from .imagenet_utils import preprocess_input, decode_predictions
                if (input_type == "file"):
                    try:
                        image_to_predict = image.load_img(image_input, target_size=(self.__input_image_size, self.__input_image_size))
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
                    predictiondata = decode_predictions(prediction, top=int(result_count))

                    for results in predictiondata:
                        countdown = 0
                        for result in results:
                            countdown += 1
                            prediction_results.append(str(result[1]))
                            prediction_probabilities.append(result[2] * 100)
                except:
                    raise ValueError("An error occured! Try again.")

                return prediction_results, prediction_probabilities
            elif (self.__modelType == "densenet"):

                model = self.__model_collection[0]

                from .DenseNet.densenet import preprocess_input, decode_predictions
                from .DenseNet.densenet import DenseNetImageNet121
                if (input_type == "file"):
                    try:
                        image_to_predict = image.load_img(image_input, target_size=(self.__input_image_size, self.__input_image_size))
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
                    predictiondata = decode_predictions(prediction, top=int(result_count))

                    for results in predictiondata:
                        countdown = 0
                        for result in results:
                            countdown += 1
                            prediction_results.append(str(result[1]))
                            prediction_probabilities.append(result[2] * 100)
                except:
                    raise ValueError("An error occured! Try again.")

                return prediction_results, prediction_probabilities
            elif (self.__modelType == "inceptionv3"):

                model = self.__model_collection[0]

                from imageai.Prediction.InceptionV3.inceptionv3 import InceptionV3, preprocess_input, decode_predictions

                if (input_type == "file"):
                    try:
                        image_to_predict = image.load_img(image_input, target_size=(self.__input_image_size, self.__input_image_size))
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
                    predictiondata = decode_predictions(prediction, top=int(result_count))

                    for results in predictiondata:
                        countdown = 0
                        for result in results:
                            countdown += 1
                            prediction_results.append(str(result[1]))
                            prediction_probabilities.append(result[2] * 100)
                except:
                    raise ValueError("An error occured! Try again.")

                return prediction_results, prediction_probabilities



    def predictMultipleImages(self, sent_images_array, result_count_per_image=2, input_type="file"):


        output_array = []

        for image_input in sent_images_array:

            prediction_results = []
            prediction_probabilities = []
            if (self.__modelLoaded == False):
                raise ValueError("You must call the loadModel() function before making predictions.")

            else:

                if (self.__modelType == "squeezenet"):

                    from .imagenet_utils import preprocess_input, decode_predictions
                    if (input_type == "file"):
                        try:
                            image_to_predict = image.load_img(image_input, target_size=(self.__input_image_size, self.__input_image_size))
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
                        predictiondata = decode_predictions(prediction, top=int(result_count_per_image))

                        for results in predictiondata:
                            countdown = 0
                            for result in results:
                                countdown += 1
                                prediction_results.append(str(result[1]))
                                prediction_probabilities.append(result[2] * 100)
                    except:
                        raise ValueError("An error occured! Try again.")

                    each_image_details = {}
                    each_image_details["predictions"] = prediction_results
                    each_image_details["percentage_probabilities"] = prediction_probabilities
                    output_array.append(each_image_details)

                elif (self.__modelType == "resnet"):

                    model = self.__model_collection[0]

                    from .imagenet_utils import preprocess_input, decode_predictions
                    if (input_type == "file"):
                        try:
                            image_to_predict = image.load_img(image_input, target_size=(self.__input_image_size, self.__input_image_size))
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
                        predictiondata = decode_predictions(prediction, top=int(result_count_per_image))

                        for results in predictiondata:
                            countdown = 0
                            for result in results:
                                countdown += 1
                                prediction_results.append(str(result[1]))
                                prediction_probabilities.append(result[2] * 100)
                    except:
                        raise ValueError("An error occured! Try again.")

                    each_image_details = {}
                    each_image_details["predictions"] = prediction_results
                    each_image_details["percentage_probabilities"] = prediction_probabilities
                    output_array.append(each_image_details)

                elif (self.__modelType == "densenet"):

                    model = self.__model_collection[0]

                    from .DenseNet.densenet import preprocess_input, decode_predictions
                    from .DenseNet.densenet import DenseNetImageNet121
                    if (input_type == "file"):
                        try:
                            image_to_predict = image.load_img(image_input, target_size=(self.__input_image_size, self.__input_image_size))
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
                        predictiondata = decode_predictions(prediction, top=int(result_count_per_image))

                        for results in predictiondata:
                            countdown = 0
                            for result in results:
                                countdown += 1
                                prediction_results.append(str(result[1]))
                                prediction_probabilities.append(result[2] * 100)
                    except:
                        raise ValueError("An error occured! Try again.")

                    each_image_details = {}
                    each_image_details["predictions"] = prediction_results
                    each_image_details["percentage_probabilities"] = prediction_probabilities
                    output_array.append(each_image_details)

                elif (self.__modelType == "inceptionv3"):

                    model = self.__model_collection[0]

                    from imageai.Prediction.InceptionV3.inceptionv3 import InceptionV3, preprocess_input, \
                        decode_predictions

                    if (input_type == "file"):
                        try:
                            image_to_predict = image.load_img(image_input, target_size=(self.__input_image_size, self.__input_image_size))
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
                        predictiondata = decode_predictions(prediction, top=int(result_count_per_image))

                        for results in predictiondata:
                            countdown = 0
                            for result in results:
                                countdown += 1
                                prediction_results.append(str(result[1]))
                                prediction_probabilities.append(result[2] * 100)
                    except:
                        raise ValueError("An error occured! Try again.")

                    each_image_details = {}
                    each_image_details["predictions"] = prediction_results
                    each_image_details["percentage_probabilities"] = prediction_probabilities
                    output_array.append(each_image_details)


        return output_array

