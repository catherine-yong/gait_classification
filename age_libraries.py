from libraries import *

all_classes_names = os.listdir('age_dataset')

dataset_directory = "age_dataset"

classes_list = ["child", "adult", "older_adult"]
model_output_size = len(classes_list)
