import csv
import cv2
import mediapipe as mp
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True)

directory = "C:\\Users\\MindRove_BZs\\Pictures\\05233_patient_11\\"
files = os.listdir(directory)

files = [f for f in files if f.endswith('.png')]
files = sorted(files, key=lambda x: int(x.split('_')[1]))

csv_directory = "C:\\Users\\MindRove_BZs\\Pictures\\05233_patient_11"
os.makedirs(csv_directory, exist_ok=True)

previous_depths = [0]*21 

with open(os.path.join(csv_directory, 'landmarks1.csv'), 'w', newline='') as csvfile:
    fieldnames = ['Frame'] + [f'landmark_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for i in range(0, len(files)-1, 2):
        depth_file = os.path.join(directory, files[i])
        image_file = os.path.join(directory, files[i+1])

        depth_image = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        image = cv2.imread(image_file)

        if depth_image is None or image is None:
            print(f"Error reading {depth_file} or {image_file}")
            break
        
        height, width, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        if i % 1000 == 0:
            print(f"Processing frame {i}...")
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = {'Frame': image_file.split('_')[4]}
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    if cx >= width:
                        cx = width - 1
                    if cy >= height:
                        cy = height - 1
                        
                    depth = depth_image[cy, cx]
                    if depth == 0:
                        depth = previous_depths[id]  
                    else:
                        previous_depths[id] = depth  

                    landmarks.update({
                        f'landmark_{id}_x': cx,
                        f'landmark_{id}_y': cy,
                        f'landmark_{id}_z': depth
                    })

                    #cv2.putText(image, f'{id}: {depth}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                writer.writerow(landmarks)

        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #cv2.imshow("Image with Landmarks", image)
        #cv2.waitKey(10)

hands.close()
cv2.destroyAllWindows()