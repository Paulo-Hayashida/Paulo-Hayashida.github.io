import cv2
import numpy as np

# def webcam():
    
#     if not cap.isOpened():
#         print("Cannot open camera")
#         exit()

#     cap = cv2.VideoCapture(0)

#     # Get current width of frame
#     width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
#     # Get current height of frame
#     height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
#     # Define Video Frame Rate in fps
#     fps = 10.0

#     controle = 'entrada'

#     print('Com a câmera ligada digite a letra "v" para gravar a imagem do pote vazio e "t" para o pote de teste. Caso tenha as imagens aperte "p" para proseguir. "Aperte "d" para usar as imagens padrões"')
#     while cap.isOpened() and controle != "saida":
    
#         k = cv2.waitKey(1)
        
#         ret, frame = cap.read()
#         if not ret:
#             print("Can't receive frame (stream end?). Exiting ...")
#             break
    
#         cv2.imshow('frame', frame)
#         if k == ord('v'):
#             cv2.imwrite('img_projeto/pote-vazio.jpg', frame)
#             print("Salvando imagem do pote vazio")
        
#         if k == ord('t'):
#             cv2.imwrite('img_projeto/pote-teste.jpg', frame)
#             print("Salvando imagem do pote para teste")
        
#         if k == ord('d'):
#             print('inciando programa default')
#             cap.release()
#             cv2.destroyAllWindows()
#             escolha = "default"
#             return escolha
            
            
#         if k == ord('p'):
#             print("Iniciando programa")
#             controle = "saida"
            

#     # Release everything if job is finished
#     cap.release()
#     cv2.destroyAllWindows()

def recortar_imagem_circular(imagem, circulo):
    x, y, r = circulo
    # Cria uma máscara circular
    mask = np.zeros_like(imagem)
    

    cv2.circle(mask, (x, y), r, (255, 255, 255), thickness=-1)
    
    # Aplica a máscara à imagem original
    imagem_cortada = cv2.bitwise_and(imagem, mask)
    mask = cv2.bitwise_not(mask)

    # Recorta a imagem no bounding box do círculo
    #imagem_cortada = imagem_cortada[y-r-5:y+r-5, x-r-5:x+r-5]
    imagem_cortada = imagem_cortada[y-r:y+r, x-r:x+r]

    
    return imagem_cortada

# def definir_cor_fundo(imagem_pote_vazio, raio_buffer=5, salvar_imagem=False):
#     # Carrega a imagem do pote vazio
#     img = cv2.imread(imagem_pote_vazio)
    
#     # Detecta o círculo do pote usando a Transformada de Hough
#     circulo = detectar_circulo_pote(img, ajustar_para_borda_interna=False)

#     if circulo is not None:
#         x, y, r = circulo
        
#         # Recorta a imagem circularmente ao redor do círculo detectado
#         img_cortada = recortar_imagem_circular(img, circulo)

        
#         # Converte a imagem recortada para o espaço de cores HSV
#         hsv = cv2.cvtColor(img_cortada, cv2.COLOR_BGR2HSV)
        
#         # Cria uma máscara circular para focar apenas na área do pote
#         mask = np.zeros_like(hsv[:,:,0])

#         mask = cv2.bitwise_not(mask)


#         cv2.circle(mask, (r, r), r - raio_buffer, (255), thickness=-1)
        
#         # Aplica a máscara e calcula a cor média do fundo do pote
#         masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)

#         cor_fundo_hsv = np.median(masked_hsv[mask > 0].reshape(-1, 3), axis=0)
        
#         # Salvar imagem destacando o círculo do pote, se necessário
#         if salvar_imagem:
#             img_circulo_destacado = img.copy()
#             cv2.circle(img_circulo_destacado, (x, y), r, (0, 255, 0), 3)
#             cv2.imwrite('pote_vazio_circulo_detectado.jpg', img_circulo_destacado)
#             cv2.imwrite('imagem_cortada_pote_vazio.jpg', img_cortada)
        
#         return cor_fundo_hsv, (r, r, r)
#     else:
#         raise Exception("Não foi possível detectar o círculo do pote na imagem do pote vazio.")

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
    
    # cv2.imshow('nivel racao', hsv)
    # cv2.waitKey(0)
    
    # Define os limites inferior e superior para a cor do fundo do pote
    #lower_bound = np.array([0, 0, 168])
    #upper_bound = np.array([172, 111, 255])
    lower_bound = np.array(cor_fundo_hsv[0])
    upper_bound = np.array(cor_fundo_hsv[1])
    
    # Cria uma máscara que identifica os pixels correspondentes à cor do fundo
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # cv2.imshow('nivel racao', mask)
    # cv2.waitKey(0)
    
    # Inverter a máscara para detectar a área preta    
    #mask = cv2.bitwise_not(mask)
    
    #cv2.imshow('nivel racao', mask)
    #cv2.waitKey(0)
    
    # Salvar máscara da cor do fundo, se necessário
    #if salvar_imagem:
    #    cv2.imwrite(f'mascara_cor_fundo{imagem}.jpg', mask)
    
    # Encontra os contornos na máscara filtrada
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Salvar imagem destacando a área do fundo encontrada, se necessário
    if salvar_imagem:
        img_fundo_destacado = img_cortada.copy()
        cv2.drawContours(img_fundo_destacado, contours, -1, (0, 0, 255), 2)
        # cv2.imwrite('area_fundo_detectada.jpg', img_fundo_destacado)
        cv2.imshow('nivel racao', img_fundo_destacado)
        cv2.waitKey(0)
    	
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
#imagem_pote_vazio = 'img_projeto/5.png'
#imagem_pote_cheio = 'img_projeto/1.png'
#imagem_pote = 'img_projeto/3.png'

#escolha = str(webcam())

imagem_pote_vazio = '../img_projeto/pote-azul-vazio.jpeg'
imagem_pote = '../img_projeto/pote-azul-70.jpeg'

# escolha = ""

# if (escolha == "default"):
#     imagem_pote_vazio = 'img_projeto/pote-azul-vazio.jpeg'
#     imagem_pote = 'img_projeto/pote-azul-70.jpeg'
# else:
#    imagem_pote_vazio = 'img_projeto/pote-vazio.jpg'
#    imagem_pote = 'img_projeto/pote-teste.jpg'

#imagem_pote_vazio = 'img_projeto/pote-verde-vazio.jpg'
# cor_fundo_hsv, _ = definir_cor_fundo(imagem_pote_vazio, salvar_imagem=True)

#webcam(pote-teste)
# cor = str(input('Digite a cor desejada(branco, vermelho, azul ou verde)\n'))

# if cor == 'azul':
# 	cor_fundo_hsv = [[66, 108, 0],[155, 255, 255]]	
# elif cor == 'branco':
# 	cor_fundo_hsv = [[0, 0, 168],[172, 111, 255]]
# elif cor == 'verde':
# 	cor_fundo_hsv = [[37, 76, 209],[85, 111, 255]]
# elif cor == 'vermelho':
# 	cor_fundo_hsv = [[165, 27, 0],[179, 237, 255]]
# elif cor == 'marrom':
cor_fundo_hsv = [[5, 100, 20], [30, 255, 200]]


#imagem_pote = 'img_projeto/pote_teste.jpg'

#imagem_pote_cheio = 'img_projeto/pote-verde-cheio.jpg'
#area_threshold = 0.5  # 20% da área do pote visível

# salvar_imagem=True para salvar na pasta atual os resultados
area_vazio = detectar_racao(imagem_pote_vazio, cor_fundo_hsv, salvar_imagem=True)
area_racao = detectar_racao(imagem_pote, cor_fundo_hsv, salvar_imagem=True)

area_threshold = area_vazio * 0.50

print(f'Area de vazia: {area_racao}')
print(f'Tamanho do pote: {area_vazio}')
print(f'Valor limite: {area_threshold}')

# # Verifica se a área do fundo visível é maior que o threshold
# if area_racao < area_threshold:
#     print("Tem ração " + str(100*(1-(area_racao/area_vazio))) + '%')
# else:
#     print("Abastecer " + str(100*(1-(area_racao/area_vazio))) + '%')