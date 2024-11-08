# pneumonia-prediction
 Pneumonia Detection app: Upload X-ray, predict normal/pneumonia using pre-trained model, display result. Quick, accurate tool for healthcare diagnosis.

The Pneumonia Detection application is a GUI-based tool that applies a convolutional neural network to predict whether a given X-ray image belongs to a regular patient or a patient with pneumonia. 


1. User uploads an X-ray image through the GUI interface.
2. The application preprocesses the image and converts it into a 3-channel RGB format. It then passes the processed image to the pre-trained deep learning model.
3. The model predicts the class probabilities (0 for normal, 1 for pneumonia) based on the image features learned during training.
4. The application displays the prediction result and the X-ray image on the GUI, enabling users to interpret and diagnose the patient's condition.

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia (Source data)
