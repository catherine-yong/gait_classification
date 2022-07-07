
from test import * 
from videos import * 

# record and save video
record_gait()

# make prediction
make_average_predictions('videos/to_predict.mp4' , 50)

print("Finished ...")