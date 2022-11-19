import cv2 as cv2

img = cv2.imread(<image path>)

dh, dw, _ = img.shape
print(dh,dw)

x,y,w,h = 0.360667, 0.089000, 0.113333, 0.130000

x,y,w,h = int(x*dw), int(y*dh), int(w*dw), int(h*dh)

print(x, y, w, h)

imgCrop = img[y:y+h,x:x+w]


cv2.imshow("Crop Image",imgCrop)