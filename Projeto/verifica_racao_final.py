import cv2
import numpy as np

def recortar_imagem_circular(imagem, circulo):
    x, y, r = circulo
    # Cria uma máscara circular
    mask = np.zeros_like(imagem)
    

    cv2.circle(mask, (x, y), r, (255, 255, 255), thickness=-1)
    
    # Aplica a máscara à imagem original
    imagem_cortada = cv2.bitwise_and(imagem, mask)
    mask = cv2.bitwise_not(mask)

    # Recorta a imagem no bounding box do círculo
    imagem_cortada = imagem_cortada[y-r-5:y+r-5, x-r-5:x+r-5]

    
    return imagem_cortada

def definir_cor_fundo(imagem_pote_vazio, raio_buffer=5, salvar_imagem=False):
    # Carrega a imagem do pote vazio
    img = cv2.imread(imagem_pote_vazio)
    
    # Detecta o círculo do pote usando a Transformada de Hough
    circulo = detectar_circulo_pote(img, ajustar_para_borda_interna=False)

    if circulo is not None:
        x, y, r = circulo
        
        # Recorta a imagem circularmente ao redor do círculo detectado
        img_cortada = recortar_imagem_circular(img, circulo)

        
        # Converte a imagem recortada para o espaço de cores HSV
        hsv = cv2.cvtColor(img_cortada, cv2.COLOR_BGR2HSV)
        
        # Cria uma máscara circular para focar apenas na área do pote
        mask = np.zeros_like(hsv[:,:,0])

        mask = cv2.bitwise_not(mask)


        cv2.circle(mask, (r, r), r - raio_buffer, (255), thickness=-1)
        
        # Aplica a máscara e calcula a cor média do fundo do pote
        masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)

        cor_fundo_hsv = np.median(masked_hsv[mask > 0].reshape(-1, 3), axis=0)
        
        # Salvar imagem destacando o círculo do pote, se necessário
        if salvar_imagem:
            img_circulo_destacado = img.copy()
            cv2.circle(img_circulo_destacado, (x, y), r, (0, 255, 0), 3)
            cv2.imwrite('pote_vazio_circulo_detectado.jpg', img_circulo_destacado)
            cv2.imwrite('imagem_cortada_pote_vazio.jpg', img_cortada)
        
        return cor_fundo_hsv, (r, r, r)
    else:
        raise Exception("Não foi possível detectar o círculo do pote na imagem do pote vazio.")

def detectar_racao(imagem, cor_fundo_hsv, raio_buffer=5, salvar_imagem=False):
    # Carrega a imagem
    img = cv2.imread(imagem)
    
    # Detecta o círculo do pote usando a Transformada de Hough
    circulo_pote = detectar_circulo_pote(img, ajustar_para_borda_interna=True)
    if circulo_pote is None:
        raise Exception("Não foi possível detectar o círculo do pote na imagem.")
    
    # Recorta a imagem circularmente ao redor do círculo detectado
    img_cortada = recortar_imagem_circular(img, circulo_pote)


    
    # Converte a imagem recortada para o espaço de cores HSV
    hsv = cv2.cvtColor(img_cortada, cv2.COLOR_BGR2HSV)
    
    # Define os limites inferior e superior para a cor do fundo do pote
    lower_bound = np.array([cor_fundo_hsv[0] - 10, 50, 50])
    upper_bound = np.array([cor_fundo_hsv[0] + 10, 255, 255])
    
    # Cria uma máscara que identifica os pixels correspondentes à cor do fundo
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Inverter a máscara para detectar a área preta
    mask = cv2.bitwise_not(mask)
    
    # Salvar máscara da cor do fundo, se necessário
    #if salvar_imagem:
    #    cv2.imwrite(f'mascara_cor_fundo{imagem}.jpg', mask)
    
    # Encontra os contornos na máscara filtrada
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Salvar imagem destacando a área do fundo encontrada, se necessário
    if salvar_imagem:
        img_fundo_destacado = img_cortada.copy()
        cv2.drawContours(img_fundo_destacado, contours, -1, (0, 0, 255), 2)
        cv2.imwrite('area_fundo_detectada.jpg', img_fundo_destacado)
    
    # Calcula a área total dos contornos
    area_racao = sum(cv2.contourArea(contour) for contour in contours)

    return area_racao
    

def detectar_circulo_pote(img, ajustar_para_borda_interna=False):
    # Converte para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplica um desfoque para suavizar a imagem
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detecta círculos usando a Transformada de Hough
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=30, minRadius=50, maxRadius=0)
    
    # Verifica se encontrou algum círculo
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Assume que o primeiro círculo detectado é o pote
        x, y, r = circles[0]
        
        # Ajuste para a borda interna
        if ajustar_para_borda_interna:
            r = int(r * 0.85)  # Reduz o raio para focar na borda interna
            
        return (x, y, r)
    else:
        return None
    

# Exemplo de uso
imagem_pote_vazio = 'img_projeto/pote-azul-vazio.jpeg'
cor_fundo_hsv, _ = definir_cor_fundo(imagem_pote_vazio, salvar_imagem=True)

imagem_pote = 'img_projeto/pote-azul-70.jpeg'

imagem_pote_cheio = 'img_projeto/pote-azul-70.jpeg'
#area_threshold = 0.5  # 20% da área do pote visível
area_cheio = detectar_racao(imagem_pote_cheio, cor_fundo_hsv, salvar_imagem=False)
area_racao = detectar_racao(imagem_pote, cor_fundo_hsv, salvar_imagem=False)


area_threshold = area_cheio * 0.50

print(f'Area de ração: {area_racao}')
print(f'Area pote cheio: {area_cheio}')
print(f'Valor limite: {area_threshold}')



# Verifica se a área do fundo visível é maior que o threshold
if area_racao > area_threshold:
    print("Tem ração")
else:
    print("Abastecer")



