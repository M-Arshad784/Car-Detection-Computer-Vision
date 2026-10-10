import cv2

cap = cv2.VideoCapture('../2_Intro_to_videos/car.avi')

while True:
    success, img = cap.read()
    if not  success:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Load Data
    car = cv2.CascadeClassifier('../2_Intro_to_videos/cars.xml')
    # Apply Detection
    car = car.detectMultiScale(gray, 1.2, 3)

    for (x, y, w, h) in car:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 244), 3)

    cv2.imshow('Car Detection', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()