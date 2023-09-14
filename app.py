import cv2
import pickle
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('./best_model.h5')

cap = cv2.VideoCapture(0) #capture video de la webcam
# Ajuster la résolution de la webcam
cap.set(3, 640)  # Largeur
cap.set(4, 480)  # Hauteur


while True:
    # Lire l'entrée de la webcam
    ret, frame = cap.read()
    # Redimensionner l'image pour correspondre à la forme d'entrée attendue par le modèle
    resized_frame = cv2.resize(frame, (224, 224))
    resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
    # Prétraiter l'image (par exemple, normaliser les pixels)
    preprocessed_frame = resized_frame / 255.0
    # Ajouter une dimension de lot (batch) à l'image
    batched_frame = np.expand_dims(preprocessed_frame, axis=0)
    
    #prédictions avec le modèle
    predictions = model.predict(batched_frame)
    if predictions < 0.5:
        print('not wearing a mask.-',predictions)
    else:
        print('wearing a mask. -',predictions )
    
    
    # Afficher le flux en direct
    cv2.imshow('Webcam', frame)
    
    time.sleep(0.5) # temps d'attempte en seconde pour evité de faire frire le cpu
    
    # Quitter la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Libérer la webcam et fermer les fenêtres d'affichage
cap.release()
cv2.destroyAllWindows()
