import cv2
import mediapipe as mp
import numpy as np
import time

# Tipagem 
confidence = float
webcam_image = np.ndarray
rgb_tuple = tuple[int, int, int]

# Classe 
class Detector:
    def __init__(self, mode: bool=False,
                 number_hands: int = 2,
                 model_complexity: int = 1,
                 min_detec_confidence: confidence = 0.5,
                 min_tracking_confidence: confidence = 0.5):
        
        # Parametros necessarios para inicializar o hands
        self.mode = mode
        self.max_num_hands = number_hands
        self.complexity = model_complexity
        self.detection_con = min_detec_confidence
        self.tracking_con = min_tracking_confidence
        
        # Inicializar o Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode,
                                        self.max_num_hands,
                                        self.complexity,
                                        self.detection_con,
                                        self.tracking_con)
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]
        
    def find_hands(self,
                   img: webcam_image,
                   draw_hands: bool = True):
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Coletar resultados do processo das hands e analisar
        self.results = self.hands.process(img_RGB)
        
        if self.results.multi_hand_landmarks and draw_hands:
            for hand in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand, self.mp_hands.HAND_CONNECTIONS)

        return img
    
    def find_position(self,
                       img: webcam_image,
                       hand_number: int = 0):
        self.required_landmark_list = []

        if self.results.multi_hand_landmarks:
            height, width = img.shape[:2]
            my_hand = self.results.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(my_hand.landmark):
                center_x, center_y = int(lm.x*width), int(lm.y*height)
                
                self.required_landmark_list.append([id, center_x, center_y])
            
            
        return self.required_landmark_list

# Teste de classe 
if __name__ == '__main__':
    # Coletando o FPS
    previous_time = 0
    current_time = 0 
    # Classe
    Detec = Detector()

    # Captura de imagem
    capture = cv2.VideoCapture(0)
    while not (cv2.waitKey(20) & 0xFF == ord('q')):
        
        # Captura do frame
        _, img = capture.read()

        # Manipulacao de frame
        img = Detec.find_hands(img)
        landmark_list = Detec.find_position(img)
        if landmark_list:
           desc1 = f"Polegar: {landmark_list[4][1:]}    Indicador: {landmark_list[8][1:]}    Medio: {landmark_list[12][1:]}"
           desc2 = f"Anelar: {landmark_list[16][1:]}    Mindinho: {landmark_list[20][1:]}"
           cv2.putText(img, desc1, (10, (img.shape[0] - 35)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1)
           cv2.putText(img, desc2, (10, (img.shape[0] - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1)

        # Calculando FPS
        current_time = time.time()
        fps = 1/(current_time - previous_time)
        previous_time = current_time
        
        #Mostrando imagem
        cv2.putText(img, str(int(fps)), (10, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2)
        cv2.imshow('Camera', img)
        
   
        
