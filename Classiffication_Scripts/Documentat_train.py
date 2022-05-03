from utils.models.Custom import ModelTraining
import os
from PIL import ImageFile
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-d', '--data',  help='data_directory')
parser.add_argument("-n",  "--nobj", default="", help="number of classes")
parser.add_argument('-e', '--epoch', default="100", help='number of epochs')
parser.add_argument('-s', '--batch', default="4", help='batch size')
parser.add_argument('-m', '--model',  default='', help='model path for extraction')
parser.add_argument('-o', '--option', default='', help='run from base or incremental')

args = parser.parse_args()
input_data = args.data
num_classes = args.nobj
num_epochs =args.epoch
bt_size =args.batch
base_model =args.model
opt_model=args.option


ImageFile.LOAD_TRUNCATED_IMAGES = True
trainer = ModelTraining()
trainer.setModelTypeAsInceptionV3()
trainer.setDataDirectory(str(input_data))
if(opt_model=="b"):
       trainer.trainModel(num_objects=int(num_classes), num_experiments=int(num_epochs), enhance_data=True, batch_size=int(bt_size), show_network_summary=True, transfer_from_model=str(base_model), initial_num_objects=1000)
elif(opt_model=="i"): 
	 trainer.trainModel(num_objects=int(num_classes), num_experiments=int(num_epochs), enhance_data=True, batch_size=int(bt_size), show_network_summary=True, continue_from_model=str(base_model), initial_num_objects=1000)