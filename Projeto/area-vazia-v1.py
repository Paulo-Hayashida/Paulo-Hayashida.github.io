import cv2
import numpy as np

def definir_cor_fundo(imagem_pote_vazio):
    # Carrega a imagem do pote vazio
    img = cv2.imread(imagem_pote_vazio)
    
    # Converte a imagem para o espaço de cores HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Calcula a média da cor na imagem
    cor_fundo_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
    
    # Retorna a cor do fundo em HSV
    return cor_fundo_hsv

def detectar_racao(imagem, cor_fundo_hsv, area_threshold):
    # Carrega a imagem
    img = cv2.imread(imagem)
    
    # Converte a imagem para o espaço de cores HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define os limites inferior e superior para a cor do fundo do pote
    lower_bound = np.array([cor_fundo_hsv[0] - 10, 50, 50])
    upper_bound = np.array([cor_fundo_hsv[0] + 10, 255, 255])
    
    # Cria uma máscara que identifica os pixels correspondentes à cor do fundo
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Encontra os contornos na máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calcula a área total dos contornos
    area_fundo = sum(cv2.contourArea(contour) for contour in contours)
    
    # Calcula a área total da imagem
    area_imagem = img.shape[0] * img.shape[1]
    
    # Verifica se a área do fundo visível é maior que o threshold
    if area_fundo / area_imagem > area_threshold:
        return False  # Não há ração
    else:
        return True  # Há ração

# Exemplo de uso
imagem_pote_vazio = 'caminho/para/imagem_pote_vazio.jpg'
cor_fundo_hsv = definir_cor_fundo(imagem_pote_vazio)

imagem_pote = 'caminho/para/sua/imagem.jpg'
area_threshold = 0.2  # 20% da área do pote visível
resultado = detectar_racao(imagem_pote, cor_fundo_hsv, area_threshold)

print("Tem ração?", resultado)
