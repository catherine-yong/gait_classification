from age_libraries import *

## every function dealing with video treatment

# Setting sthe Widow Size which will be used by the Rolling Averge Proces
window_size = 1
output_directory = "videos_test"
video_title = "to_predict"

def frames_extraction(video_path):
    # Empty List declared to store video frames
    frames_list = []
    
    # Reading the Video File Using the VideoCapture
    video_reader = cv2.VideoCapture(video_path)

    # Iterating through Video Frames
    while True:

        # Reading a frame from the video file 
        success, frame = video_reader.read() 

        # If Video frame was not successfully read then break the loop
        if not success:
            break

        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (image_height, image_width))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255
        
        # Appending the normalized frame into the frames list
        frames_list.append(normalized_frame)
    
    # Closing the VideoCapture object and releasing all resources. 
    video_reader.release()

    # returning the frames list 
    return frames_list

def calculate_angle(a,b,c): 
    a = np.array(a) #premier
    b = np.array(b) #deuxième
    c = np.array(c) #troisième
    
    #on va faire le calcul en radian
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0]) 
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
        return angle


def record_gait():
    # enregistre une video sans Mediapipe - mais affiche en live avec Mediapipe - 
    # et la sauvegarde dans le dossier Youtube_Videos
    
    
    video = cv2.VideoCapture(0)
    counter = 0 #compteur pour le comptage des mouvements
    stage = None

    if (video.isOpened() == False):
        print("Error reading video file")

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    size = (frame_width, frame_height)

    result = cv2.VideoWriter('videos_test/to_predict.mp4',cv2.VideoWriter_fourcc(*'MJPG'),10, size)


    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while video.isOpened():
            ret, frame = video.read() ## frame : video sans aucun filtre

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) ## image : video avec mediapipe
            image.flags.writeable = False

            #Make detection
            results = pose.process(image)

            #Recolor image to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
            try :
            #Render curl counter setup statux box
                landmarks = results.pose_landmarks.landmark
                
                #get coordinates
                hip=[landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee=[landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle=[landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                #calculate knee angle 
                angle = calculate_angle(hip, knee, ankle)
                
                
                if angle < 170:
                        stage = "down"
                if angle > 170 and stage == "down":
                        stage = "up"
                        counter +=1
            except :
                pass

            cv2.rectangle(image,(0,0),(100,70),(245,117,16),-1)
            cv2.putText(image,'REPS' , (30,15),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1, cv2.LINE_AA)
            cv2.putText(image,str(counter),(30,40),cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255),2,cv2.LINE_AA)
                
            
            #Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            cv2.imshow('Gait + Mediapipe', image)

            if ret == True:
                result.write(frame) ## choisir ici pour l'image sans le squelette Mediapipe
                #result.write(image) ## on enregistre l'image avec le squelette Mediapipe
                #cv2.imshow('Gait', frame)

                if counter == 2 or cv2.waitKey(1) & 0xFF == ord('q'): ## on s'arrête si on a fait 2 pas ou arrêt forcé
                    #print("OK")
                    break

            else:
                print("KO")
                
                break


        video.release()
        result.release()
        cv2.destroyAllWindows()
        return counter