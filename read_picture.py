#初步处理照片，识别照片中的人脸，规范化成220*220jpg文件。
import cv2

XX="CM"#类型 AF AM CF CM

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

for i in range(1,751):
    print(i)
    #Read the image anf BGR to GRAY
    imagePath = "E:\\daxue\\graduation\\SCUT-FBP5500_v2\\Images\\"+XX+str(i)+".jpg"
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    for (x, y, w, h) in faces:
        new_image = image[y:y+h, x:x+w]
        print(new_image.shape)
        new2_image=cv2.resize(new_image,(220,220),interpolation=cv2.INTER_CUBIC)
        print(new2_image.shape)
        break#默认第一个
    cv2.imwrite("E:\\daxue\\graduation\\face\\" + XX + str(i) + ".jpg", new2_image)
