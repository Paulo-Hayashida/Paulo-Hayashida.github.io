import cv2
import numpy as np

cap = cv2.VideoCapture(0)

cor_fundo_hsv = [[5, 100, 20], [30, 255, 200]]

def pote_cheio():
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        font = cv2.FONT_HERSHEY_SIMPLEX 
        img_destacada, area_racao  = detectar_racao(frame, cor_fundo_hsv)
        texto_video = str(area_racao)
        cv2.putText(img_destacada, 
                'Press S save value | Press Q to quit',  
                (50, 50),  
                font, 0.7,  
                (0, 0, 255),  
                2,  
                cv2.LINE_4) 
        
        cv2.putText(img_destacada, 
                texto_video,  
                (50, 20),  
                font, 0.7,  
                (0, 255, 0),  
                2,  
                cv2.LINE_4) 

        # Display the resulting frame
        cv2.imshow('frame', img_destacada)

        # Save on "s" key or exit on "q"
        k = cv2.waitKey(1) 
        if  k == ord('s'):
            cv2.imwrite("pote_cheio.jpg",img_destacada)
            return area_racao

        elif k == ord('q'):
            return 5

    # # When everything done, release the capture
    # cap.release()
    # cv2.destroyAllWindows()


def webcam():
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()


    cor_fundo_hsv = [[5, 100, 20], [30, 255, 200]]


    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break


        font = cv2.FONT_HERSHEY_SIMPLEX 
        img_destacada, area_racao_atual  = detectar_racao(frame, cor_fundo_hsv)
        show_percent, status_pote = calcula_percent(area_racao_atual)
        texto_video = str(show_percent)
        if show_percent > 1:
            cv2.putText(img_destacada, status_pote, (50, 20),font, 1,  
                (255, 0, 20),  
                2,cv2.LINE_4)

        elif show_percent > 0.5:
            cv2.putText(img_destacada, texto_video, (50, 50),font, 1,  
                (255, 0, 0),  
                2,cv2.LINE_4)
            cv2.putText(img_destacada, status_pote, (50, 20),font, 1,  
                (255, 0, 0),  
                2,cv2.LINE_4)
        else:
            cv2.putText(img_destacada, texto_video, (50, 50),font, 1,  
                (0, 0, 255),  
                2,cv2.LINE_4)
            cv2.putText(img_destacada, status_pote, (50, 20),font, 1,  
                (0, 0, 255),  
                2,cv2.LINE_4)
        # Display the resulting frame
        cv2.imshow('img_destacada', img_destacada)
        

        # Save on "s" key or exit on "q"
        k = cv2.waitKey(1) 
        if  k == ord('s'):
            cv2.imwrite("racao_atual.jpg",img_destacada)
        elif k == ord('q'):
            break
            

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()

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

def calcula_percent(area_atual):
    if area_racao*0.5 > area_atual:
        percent = area_atual/area_racao
        return percent, "Abastecer"
    else:
        percent = area_atual/area_racao
        return percent, "Pote Cheio"



def detectar_racao(img, cor_fundo_hsv):
    # Carrega a imagem
    #img = cv2.imread(imagem)
    
    # Detecta o círculo do pote usando a Transformada de Hough
#    circulo_pote = detectar_circulo_pote(img, ajustar_para_borda_interna=True)
    # if circulo_pote is None:
    #     raise Exception("Não foi possível detectar o círculo do pote na imagem.")
    
    # Recorta a imagem circularmente ao redor do círculo detectado
#    img_cortada = recortar_imagem_circular(img, circulo_pote)
    
    # Converte a imagem recortada para o espaço de cores HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
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
    
    font = cv2.FONT_HERSHEY_SIMPLEX 
  
    # Use putText() method for 
    # inserting text on video 
  

    # Salvar imagem destacando a área do fundo encontrada, se necessário
    img_fundo_destacado = img.copy()
    cv2.drawContours(img_fundo_destacado, contours, -1, (0, 0, 255), 2)
    # cv2.imwrite('area_fundo_detectada.jpg', img_fundo_destacado)
    # cv2.putText(img_fundo_destacado, 
    #             'AUUUUUUUUUU',  
    #             (50, 50),  
    #             font, 1,  
    #             (0, 255, 255),  
    #             2,  
    #             cv2.LINE_4) 
    # cv2.imshow('nivel racao', img_fundo_destacado)
    #cv2.waitKey(0)
    	
    # Calcula a área total dos contornos
    area_racao = sum(cv2.contourArea(contour) for contour in contours)

    return img_fundo_destacado, area_racao

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

area_racao = pote_cheio()
if area_racao != 5:
    webcam()