from pretreatment import * 
from model import *
from loss_accuracy import *   

# Calling the create_model method
model = create_model()

print("Model Created Successfully !!")

# Adding the Early Stopping Callback to the model which will continuously monitor the validation loss metric for every epoch.
# If the models validation loss does not decrease after 15 consecutive epochs, the training will be stopped and the weight which reported the lowest validation loss will be retored in the model.
early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 15, mode = 'min', restore_best_weights = True)

# Adding loss, optimizer and metrics values to the model.
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])

# Start Training
model_training_history = model.fit(x = features_train, y = labels_train, epochs = 3, batch_size = 2 , shuffle = True, validation_split = 0.2, callbacks = [early_stopping_callback])



#plot_metric('loss', 'val_loss', 'Total Loss vs Total Validation Loss')
#plot_metric('accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')


