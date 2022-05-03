from all.models.Custom import CustomImagePrediction
import os
import keras

from argparse import ArgumentParser
print(keras.__version__)
class DataHelper(object):
    def __init__(self):
        pass

    def write_data_to_file(self, file, probability, prediction):

        if float(probability) >= 70.0:
            row = os.path.basename(file) + "," + prediction + "," + str(probability) + ",Pass"
        else:
            row = os.path.basename(file) + "," + prediction + "," + str(probability) + ",Fail"
        file1 = open(folder_path+"Output.csv", "a")
        file1.write(row + "\n")

    def put_file_into_specific_class(self, path, file_name, class_name):
        if os.path.exists(path + "\\" + class_name):
            os.rename(path + "\\" + file_name, path + "\\" + class_name + "\\" + file_name)
        else:
            os.mkdir(path + "\\" + class_name)
            os.rename(path + "\\" + file_name, path + "\\" + class_name + "\\" + file_name)


class Classification(object):
    def __init__(self, model_path, json_path, num_objects):
        self.prediction = CustomImagePrediction()
        self.prediction.setModelTypeAsInceptionV3()
        self.load_model(model_path, json_path, num_objects)


    def load_model(self, model_path, json_path, num_objects):
        self.prediction.setModelPath(model_path)
        self.prediction.setJsonPath(json_path)
        self.prediction.loadModel(num_objects=num_objects)

    def classify(self, image_path):
        return self.prediction.predictImage(image_path, result_count=1, input_type="file")



parser = ArgumentParser()
parser.add_argument("-m", "--model",dest="model", help="path to model ")
parser.add_argument("-l", "--labels",dest="labels", help="Relative path to labels")
parser.add_argument("-f", "--file", dest="myFile", help="Open specified file")
parser.add_argument("-n", "--num", dest="num_objects", help="No. of output classes")

args = parser.parse_args()
model_path = args.model
json_path = args.labels
folder_path = args.myFile

print(folder_path)

print(model_path)

# i=0
classifier = Classification(model_path+"\\model.h5", json_path+"\\model_class.json", num_objects=202)
print("model")
data_helper = DataHelper()
print ("out of data_helper")
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        #file_path=os.path.join(dir, filename)
        prediction, probability = classifier.classify(folder_path+"\\"+filename)
        print(prediction, probability)
        if float(probability[0]) >= 70.0:
            data_helper.write_data_to_file(folder_path + "\\" +filename, probability[0], prediction[0])
            data_helper.put_file_into_specific_class(folder_path, filename, prediction[0])
            # shutil.rmtree("temp", ignore_errors=False, onerror=None)
            #break
        else:
            data_helper.write_data_to_file(folder_path + "\\" + filename, probability[0], prediction[0])
        
            














