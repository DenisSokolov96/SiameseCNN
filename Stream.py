import datetime

from Network import *


class Stream:
    @classmethod
    def start(cls, width, height, model):
        faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        # Включаем первую камеру
        cap = cv2.VideoCapture(0)
        cap.set(3, 480)  # set Width
        cap.set(4, 640)  # set Height
        num = 0
        str = ""
        while True:
            ret, img = cap.read()
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(20, 20)
            )
            for (x, y, w, h) in faces:
                num += 1
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, str, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                if num == 20:
                    num = 0
                    gray = cv2.resize(gray, (width, height),  interpolation=cv2.INTER_AREA)
                    #now = datetime.datetime.now().strftime("%d-%m-%Y-(%H-%M)")
                    #cv2.imwrite("image/IMG " + now + ".jpg", gray)
                    Network.test(gray, model)

            cv2.imshow('Camera', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Камера отключена.")
