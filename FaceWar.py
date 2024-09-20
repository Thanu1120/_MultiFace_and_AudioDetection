import cv2
import pyaudio
import numpy as np

#Start..............

# Parameters
CHUNK = 1024  # Number of audio samples per buffer
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Mono audio
RATE = 44100  # Sampling rate in Hz
THRESHOLD = 500  # Volume threshold for warning

#Endd..................


#Start....................

# Initialize PyAudio
p = pyaudio.PyAudio()
# Open a stream for audio input
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Listening...")

#End.....................

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the video capture from the default camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    #Start.............

    # Read audio data from the stream
    data = stream.read(CHUNK)
    audio_data = np.frombuffer(data, dtype=np.int16)

    # Calculate the volume
    volume = np.linalg.norm(audio_data)

    # Check if volume exceeds the threshold
    if volume > THRESHOLD:
        print("Warning: High audio detected!")
        height, width = frame.shape[:2] 
        cv2.putText(frame, "WARNING: Audio Detected..............", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    #Endd............

    
    
    # Convert the frame to grayscale (Haar Cascade works better with grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Check if more than one face is detected
    if len(faces) > 1:
        cv2.putText(frame, "WARNING: Multiple Faces Detected!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    elif len(faces) == 1:
        cv2.putText(frame, "One Face Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "No Face Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Face Detection', frame)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the stream and terminate PyAudio
stream.stop_stream()
stream.close()
p.terminate()

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
