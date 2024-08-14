import cv2
import numpy as np

def webcam(cor_fundo_hsv, circulo_pote, raio_buffer=5):

	cap = cv2.VideoCapture(0)

	width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	
	while cap.isOpened():
		# Capture frame-by-frame
		ret, frame = cap.read()
		# if frame is read correctly ret is True
		if not ret:
			print("Can't receive frame (stream end?). Exiting ...")
			break
    
		# Converte a imagem para o espaço de cores HSV
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		# Define os limites inferior e superior para a cor do fundo do pote
		lower_bound = np.array([cor_fundo_hsv[0] - 10, 50, 50])
		upper_bound = np.array([cor_fundo_hsv[0] + 10, 255, 255])

		# Cria uma máscara que identifica os pixels correspondentes à cor do fundo
		mask = cv2.inRange(hsv, lower_bound, upper_bound)

		# Extrai o círculo do pote para focar a análise
		x, y, r = circulo_pote
		mask_circle = np.zeros_like(mask)
		cv2.circle(mask_circle, (x, y), r - raio_buffer, (255), thickness=-1)

		# Aplica a máscara circular à máscara da cor do fundo
		masked_area = cv2.bitwise_and(mask, mask, mask=mask_circle)

		# Encontra os contornos na máscara filtrada
		contours, _ = cv2.findContours(masked_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		img_fundo_destacado = frame.copy()
		cv2.drawContours(img_fundo_destacado, contours, -1, (0, 0, 255), 2)
		#cv2.imwrite('area_fundo_detectada.jpg', img_fundo_destacado)
		# Display the resulting frame
		cv2.imshow('frame', img_fundo_destacado)

		if cv2.waitKey(1) == ord('q'):
		    	break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

def definir_cor_fundo(imagem_pote_vazio, raio_buffer=5, salvar_imagem=False):
    # Carrega a imagem do pote vazio
    img = cv2.imread(imagem_pote_vazio)
    
    # Converte a imagem para o espaço de cores HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Detecta o círculo do pote usando a Transformada de Hough
    circulo = detectar_circulo_pote(img)

    if circulo is not None:
        x, y, r = circulo
        
        # Cria uma máscara circular para focar apenas na área do pote
        mask = np.zeros_like(hsv[:,:,0])
        cv2.circle(mask, (x, y), r - raio_buffer, (255), thickness=-1)
        
        # Aplica a máscara e calcula a cor média do fundo do pote
        masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
        cor_fundo_hsv = np.mean(masked_hsv[mask > 0].reshape(-1, 3), axis=0)
        
        # Salvar imagem destacando o círculo do pote, se necessário
        if salvar_imagem:
            img_circulo_destacado = img.copy()
            cv2.circle(img_circulo_destacado, (x, y), r, (0, 255, 0), 3)
            cv2.imwrite('pote_vazio_circulo_detectado.jpg', img_circulo_destacado)
        
        return cor_fundo_hsv, (x, y, r)
    else:
        raise Exception("Não foi possível detectar o círculo do pote na imagem do pote vazio.")

def detectar_racao(imagem, cor_fundo_hsv, circulo_pote, area_threshold, raio_buffer=5, salvar_imagem=False):
    # Carrega a imagem
    img = cv2.imread(imagem)
    
    # Converte a imagem para o espaço de cores HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define os limites inferior e superior para a cor do fundo do pote
    lower_bound = np.array([cor_fundo_hsv[0] - 10, 50, 50])
    upper_bound = np.array([cor_fundo_hsv[0] + 10, 255, 255])
    
    # Cria uma máscara que identifica os pixels correspondentes à cor do fundo
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Extrai o círculo do pote para focar a análise
    x, y, r = circulo_pote
    mask_circle = np.zeros_like(mask)
    cv2.circle(mask_circle, (x, y), r - raio_buffer, (255), thickness=-1)
    
    # Aplica a máscara circular à máscara da cor do fundo
    masked_area = cv2.bitwise_and(mask, mask, mask=mask_circle)
    
    # Encontra os contornos na máscara filtrada
    contours, _ = cv2.findContours(masked_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Teste
    cv2.imshow('masked', masked_area)
    cv2.waitKey(0)
    
    # Salvar imagem destacando a área do fundo encontrada, se necessário
    if salvar_imagem:
        img_fundo_destacado = img.copy()
        cv2.drawContours(img_fundo_destacado, contours, -1, (0, 0, 255), 2)
        cv2.imwrite('area_fundo_detectada.jpg', img_fundo_destacado)
        cv2.imshow('fundo', img_fundo_destacado)
    
    # Calcula a área total dos contornos
    area_fundo = sum(cv2.contourArea(contour) for contour in contours)
    
    # Calcula a área total do círculo do pote
    area_pote = np.pi * (r - raio_buffer) ** 2
    
    # Verifica se a área do fundo visível é maior que o threshold
    if area_fundo / area_pote > area_threshold:
        return False  # Não há ração
    else:
        return True  # Há ração

def detectar_circulo_pote(img):
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
        return circles[0]
    else:
        return None

# Exemplo de uso
# imagem_pote_vazio = 'caminho/para/imagem_pote_vazio.jpg'
# cor_fundo_hsv, circulo_pote = definir_cor_fundo(imagem_pote_vazio, salvar_imagem=True)

# imagem_pote = 'caminho/para/sua/imagem.jpg'
# area_threshold = 0.2  # 20% da área do pote visível
#resultado = detectar_racao(imagem_pote, cor_fundo_hsv, circulo_pote, area_threshold, salvar_imagem=True)

imagem_pote_vazio = './5.png'
cor_fundo_hsv, circulo_pote = definir_cor_fundo(imagem_pote_vazio, salvar_imagem=True)

imagem_pote = './3.png'
area_threshold = 0.2  # 20% da área do pote visível
#resultado = detectar_racao(imagem_pote, cor_fundo_hsv, circulo_pote, area_threshold, salvar_imagem=True)

webcam(cor_fundo_hsv, circulo_pote)

print("Tem ração?", resultado)
