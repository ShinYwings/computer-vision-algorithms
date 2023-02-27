```python
clf = joblib.load("person_final.pkl")
orig = cv2.imread("pedestrian.jpeg")
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

scaleFactor = 1.5
inverse = 1.0/scaleFactor
winStride = (8, 8)
winSize = (128, 64)
rects = []

h, w = gray.shape
count = 0
while (h >= 128 and w >= 64):
    h, w= gray.shape
    horiz = w - 64
    vert = h - 128
    i = 0
    j = 0
    while i < vert:
        j = 0
        while j < horiz:
            portion = gray[i:i+winSize[0], j:j+winSize[1]]
            features = hog(portion)
            result = clf.predict([features])
            if int(result[0]) == 1:
                
                confidence = clf.decision_function([features])
                appendRects(i, j, confidence, count, rects)
            j = j + winStride[0]
        i = i + winStride[1]
    count = count + 1

nms_rects = nms(rects, 0.2)

for (a, b, conf, c, d) in rects:
    cv2.rectangle(orig, (a, b), (a+c, b+d), (0, 255, 0), 2)

cv2.imshow("Before NMS", orig)
cv2.waitKey(0)

for (a, b, conf, c, d) in nms_rects:
    cv2.rectangle(img, (a, b), (a+c, b+d), (0, 255, 0), 2)

cv2.imshow("After NMS", img)
cv2.waitKey(0)
```