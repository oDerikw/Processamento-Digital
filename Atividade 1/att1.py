import cv2 #importa a biblioteca
import numpy as np # log
from matplotlib import pyplot as plt # funções não disponiveis em cv

img1 = cv2.imread('lena.png') #abrir a imagem colorida
img2 = cv2.imread('img_aluno.jpg') #abrir a imagem cinza
img3 = cv2.imread('unequalized.jpg') #abrir a imagem cinza

# 1 Conversão para cinza
(canalAzul1, canalVerde1, canalVermelho1) = cv2.split(img1) # tons de cinza
(canalAzul2, canalVerde2, canalVermelho2) = cv2.split(img2) # tons de cinza
cv2.imshow("Cinza", canalVerde1) # Monstra a img
cv2.waitKey(0)
cv2.imshow("Cinza2", canalVerde2) # Monstra a img
cv2.waitKey(0)
cv2.imwrite("lena-cinza.jpg", canalVerde1) #Salvar a imagem no disco com função imwrite()
cv2.imwrite("aluno-cinza.jpg", canalVerde2) #Salvar a imagem no disco com função imwrite()
cv2.waitKey(0)

# 2 Imagem Inversa
img_neg1 = 255 - img1 # Subtrai valor das imgs
img_neg2 = 255 - img2 # Subtrai valor das imgs
cv2.imshow('negative1', img_neg1)
cv2.waitKey(0)
cv2.imshow('negative2', img_neg2)
cv2.waitKey(0)
cv2.imwrite("lena-invertido.jpg", img_neg1) #Salvar a imagem no disco com função imwrite()
cv2.imwrite("aluno-invertido.jpg", img_neg2) #Salvar a imagem no disco com função imwrite()

# 3 Normalização usando = (50,150)
norm = np.zeros((800,800))
img_normalized1 = cv2.normalize(img1,  norm, 50, 150, cv2.NORM_MINMAX)
img_normalized2 = cv2.normalize(img2,  norm, 50, 150, cv2.NORM_MINMAX)
cv2.imshow('normal1', img_normalized1)
cv2.waitKey(0)
cv2.imshow('normal2', img_normalized2)
cv2.waitKey(0)
cv2.imwrite("lena-normal.jpg", img_normalized1) #Salvar a imagem no disco com função imwrite()
cv2.imwrite("aluno-normal.jpg", img_normalized2) #Salvar a imagem no disco com função imwrite()

# 4 Aplicar Logaritmo
c = (255/(np.log (1+255))) # Aplica log
img_log1 = c * np.log((1+img1.astype(np.int32)))
img_log1 = np.array(img_log1,dtype=np.uint8)

img_log2 = c * np.log((1+img2.astype(np.int32)))
img_log2 = np.array(img_log2,dtype=np.uint8)

cv2.imshow('log1', img_log1)
cv2.waitKey(0)
cv2.imshow('log2', img_log2)
cv2.waitKey(0)
cv2.imwrite("aluno-normal.jpg", img_log2) #Salvar a imagem no disco com função imwrite()
cv2.imwrite("lena-normal.jpg", img_log1) #Salvar a imagem no disco com função imwrite()

# 5 Operador Logistico
k = 0.050
img_op1 = (255/(1+np.exp(-k*(img1.astype(np.int32)-127)))).astype(np.uint8)
img_op2 = (255/(1+np.exp(-k*(img2.astype(np.int32)-127)))).astype(np.uint8)
plt.subplot(121)
plt.imshow(cv2.cvtColor(img_op1, cv2.COLOR_BGR2RGB),cmap="gray") 
plt.axis('off')
plt.subplot(122)
plt.imshow(cv2.cvtColor(img_op2, cv2.COLOR_BGR2RGB),cmap="gray")
plt.axis('off')
plt.show()

# 6 Histograma

#Definição das funções de histograma e histograma normal.
def histograma (A, nbins):
    N, M = A.shape
    h = np.zeros(nbins).astype(int)
    for y in range(N): 
        for x in range(M):
            h[A[y, x]] +=1
    return h

def histograma_normal (A, nbins):
    N, M = A.shape
    hist = histograma(A, nbins)
    
    return hist / (N * M)

# 6 (i) - Histogramas

# Histograma da imagem 3 -> unequalized
img3_cinza = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
hist_une = histograma(img3_cinza, 256)

# Histograma da imagem 2 -> img_aluno
img1_cinza = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
hist_aluno_norm = histograma_normal(img1_cinza, 256)

plt.subplot(121)
plt.imshow(img3, cmap='gray')


plt.subplot(122)
plt.bar(range(256), hist_une)
plt.xlabel('Intensidade')
plt.ylabel('Frequencia')

plt.show()

# 6 (ii) - Dividir R G B img_aluno
b, g, r = cv2.split(img2)
hist_r= histograma(r,256)
hist_g=histograma(g,256)
hist_b=histograma(b,256)

plt.figure(figsize=(12,12))
plt.subplot(121)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

plt.subplot(122)
plt.plot(hist_r, color='red', label='R')
plt.plot(hist_g, color='green', label='G')
plt.plot(hist_b, color='blue', label='B')
plt.xlim([0, 256])
plt.xlabel('Intensidade')
plt.ylabel('Frequência')
plt.title('Histograma R, G e B')
plt.legend()
plt.show()

# 6 (iii) - Dividir em uma camada e usar (a)
hist_2 = histograma(img1_cinza, 256)
plt.subplot(121)
plt.imshow(img1_cinza, cmap='gray')
plt.subplot(122)
plt.bar(range(256), hist_2)
plt.xlabel('Intensidade')
plt.ylabel('Frequencia')
plt.show()

# Definições das funções de acumulado e equalizado
def histograma_acumulado (A, nbins):
    h = histograma(A, nbins)
    for i in range(1,h.size): 
        h[i] += h[i-1]
    return h

def histograma_eq(A, nbins):
    h_acumulado = histograma_acumulado(A, nbins)
    N, M = A.shape
    for i in range(N):
        for j in range(M):
            A[i, j] = (h_acumulado[A[i, j]] - np.min(h_acumulado)) / ((N * M) - np.min(h_acumulado)) * (nbins - 1)
    return A

# Histograma equalizado
b1, g1, r1 = cv2.split(img1)
b2, g2, r2 = cv2.split(img2)

# Equalizar RGB
img_b_eq = histograma_eq(b1, 256)
img_g_eq = histograma_eq(g1, 256)
img_r_eq = histograma_eq(r1, 256)
img_eq_merge = cv2.merge([img_b_eq, img_g_eq, img_r_eq])

# Equalizar RGB img2
img2_b_eq2 = histograma_eq(b2, 256)
img2_g_eq2 = histograma_eq(g2, 256)
img2_r_eq2 = histograma_eq(r2, 256)
img2_eq_merge2 = cv2.merge([img2_b_eq2, img2_g_eq2, img2_r_eq2])

# Histograma equalizado
hist1_eq_r = histograma(img_b_eq,256)
hist1_eq_g = histograma(img_g_eq,256)
hist1_eq_b = histograma(img_r_eq,256)

#histograma equalizado img 2
hist2_eq_r = histograma(img2_b_eq2,256)
hist2_eq_g = histograma(img2_g_eq2,256)
hist2_eq_b = histograma(img2_r_eq2,256)

# Histograma equalizado por RGB
plt.subplot(121)
plt.imshow(cv2.cvtColor(img_eq_merge, cv2.COLOR_BGR2RGB))
plt.subplot(122)
plt.plot(hist1_eq_r, color='red', label='R')
plt.plot(hist1_eq_g, color='green', label='G')
plt.plot(hist1_eq_b, color='blue', label='B')
plt.xlim([0, 256])
plt.xlabel('Intensidade')
plt.ylabel('Frequência')
plt.title('Histograma R, G e B')
plt.legend()
plt.show()

# Histograma equalizado por RGB img2
plt.subplot(121)
plt.imshow(cv2.cvtColor(img2_eq_merge2, cv2.COLOR_BGR2RGB))
plt.subplot(122)
plt.plot(hist2_eq_r, color='red', label='R')
plt.plot(hist2_eq_g, color='green', label='G')
plt.plot(hist2_eq_b, color='blue', label='B')
plt.xlim([0, 256])
plt.xlabel('Intensidade')
plt.ylabel('Frequência')
plt.title('Histograma R, G e B')
plt.legend()
plt.show()

# 7 - Histograma equalizado img 3
img_eq3 = histograma_eq(img3_cinza, 256) # Equalizar o histograma
hist3_eq = histograma(img_eq3,256) # Histograma equalizado
plt.subplot(121)
plt.imshow(img_eq3, cmap='gray')
plt.subplot(122)
plt.bar(range(256), hist3_eq)
plt.xlabel('Valor da intensidade')
plt.ylabel('Frequencia')
plt.show()

cv2. destroyAllWindows() # Fecha todas as abas