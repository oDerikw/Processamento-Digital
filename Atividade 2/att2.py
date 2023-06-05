import cv2 #importa a biblioteca
import numpy as np

img1 = cv2.imread('img_aluno1.png') #abrir a imagem cinza
img2 = cv2.imread('img_aluno2.png') #abrir a imagem cinza

cv2.imshow("img1", img1)
cv2.waitKey(0)
cv2.imshow("img2", img2)
cv2.waitKey(0)

# 0) Conversão para cinza
img1_cinza = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_cinza = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Mostra as imgs
cv2.imshow("Cinza", img1_cinza)
cv2.waitKey(0)
cv2.imshow("Cinza2", img2_cinza)
cv2.waitKey(0)

# 1) Suavização pela média
img1_md = cv2.blur(img1_cinza,(5,5))
img2_md = cv2.blur(img2_cinza,(5,5))

img1_gau = cv2.GaussianBlur(img1_cinza,(3,3),0)
img2_gau = cv2.GaussianBlur(img2_cinza,(3,3),0)

cv2.imshow("Media 1", img1_md)
cv2.waitKey(0)
cv2.imshow("Media 2", img2_md)
cv2.waitKey(0)

# 2) Suavização pela média de k vizinhos
img1_mk = cv2.medianBlur(img1_cinza, ksize=5)
img2_mk = cv2.medianBlur(img2_cinza, ksize=5)

cv2.imshow('Filtro de k 1', img1_mk)
cv2.waitKey(0)
cv2.imshow('Filtro de k 2', img2_mk)
cv2.waitKey(0)

# 3) Operador Laplaciano
img1_lpc = cv2.Laplacian(img1_md, cv2.CV_16S, ksize=5)
img1_lpc_abs = cv2.convertScaleAbs(img1_lpc)

img2_lpc = cv2.Laplacian(img2_md, cv2.CV_16S, ksize=5)
img2_lpc_abs = cv2.convertScaleAbs(img2_lpc)

# Mostra as imgs
cv2.imshow('Img1 Laplace', img1_lpc_abs)
cv2.waitKey(0)
cv2.imshow('Img2 Laplace', img2_lpc_abs)
cv2.waitKey(0)

# 4) Roberts
kernelx = np.array([[1, 0], [0, -1]])
kernely = np.array([[0, 1], [-1, 0]])
img1_robertx = cv2.filter2D(img1_gau, -1, kernelx)
img1_roberty = cv2.filter2D(img1_gau, -1, kernely)
img1_roberts = cv2.addWeighted(img1_robertx, 0.5, img1_roberty, 0.5, 0)

img2_robertx = cv2.filter2D(img2_gau, -1, kernelx)
img2_roberty = cv2.filter2D(img2_gau, -1, kernely)
img2_roberts = cv2.addWeighted(img2_robertx, 0.5, img2_roberty, 0.5, 0)

cv2.imshow('Roberts 1 X Y', img1_roberts)
cv2.waitKey(0)
cv2.imshow('Roberts 2 X Y', img2_roberts)
cv2.waitKey(0)

# 5) Prewitt
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img1_prewittx = cv2.filter2D(img1_gau, -1, kernelx)
img1_prewitty = cv2.filter2D(img1_gau, -1, kernely)

img2_prewittx = cv2.filter2D(img2_gau, -1, kernelx)
img2_prewitty = cv2.filter2D(img2_gau, -1, kernely)

cv2.imshow("Prewitt 1 X Y", img1_prewittx + img1_prewitty)
cv2.waitKey(0)
cv2.imshow("Prewitt 2 X Y", img2_prewittx + img2_prewitty)
cv2.waitKey(0)

# 6) Sobel
img1_sobelx = cv2.Sobel(img1_gau,cv2.CV_8U,1,0,ksize=3)
img1_sobely = cv2.Sobel(img1_gau,cv2.CV_8U,0,1,ksize=3)
img1_sobel = img1_sobelx + img1_sobely

img2_sobelx = cv2.Sobel(img2_gau,cv2.CV_8U,1,0,ksize=3)
img2_sobely = cv2.Sobel(img2_gau,cv2.CV_8U,0,1,ksize=3)
img2_sobel = img2_sobelx + img2_sobely
 
cv2.imshow('Sobel 1 X Y', img1_sobel)
cv2.waitKey(0)
cv2.imshow('Sobel 2 X Y', img2_sobel)
cv2.waitKey(0)

cv2. destroyAllWindows() # Fecha todas as abas