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
        


        img_destacada, area_racao  = detectar_racao(frame, cor_fundo_hsv, False)
        texto_video = str(area_racao)
        img_destacada = print_tela_red(img_destacada, 'Press S save value | Press Q to quit', 50)
        img_destacada = print_tela_green(img_destacada, texto_video, 20)

        # Save on "s" key or exit on "q"
        k = cv2.waitKey(1) 
        if  k == ord('s'):
            cv2.imwrite("pote_cheio.jpg",img_destacada)
            return area_racao

        elif k == ord('q'):
            return 5        

        # Display the resulting frame
        cv2.imshow('frame', img_destacada)





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
        img_destacada, area_racao_atual  = detectar_racao(frame, cor_fundo_hsv, False)
        show_percent, status_pote = calcula_percent(area_racao_atual)
        texto_video = str(show_percent)
        if show_percent > 1:
            img_destacada = print_tela_green(img_destacada, status_pote, 20)

        elif show_percent > 0.5:
            img_destacada = print_tela_green(img_destacada, texto_video, 50)

            img_destacada = print_tela_green(img_destacada, status_pote, 20)

        else:
            img_destacada = print_tela_red(img_destacada, texto_video, 50)

            img_destacada = print_tela_red(img_destacada, status_pote, 20)

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



def calcula_percent(area_atual):
    if area_racao*0.5 > area_atual:
        percent = area_atual/area_racao
        return percent, "Abastecer"
    else:
        percent = area_atual/area_racao
        return percent, "Pote Cheio"



def detectar_racao(img, cor_fundo_hsv, read):

    if read:
        img = cv2.imread(img)
    
    # Converte a imagem recortada para o espaço de cores HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_bound = np.array(cor_fundo_hsv[0])
    upper_bound = np.array(cor_fundo_hsv[1])
    
    # Cria uma máscara que identifica os pixels correspondentes à cor do fundo
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
   
    # Encontra os contornos na máscara filtrada
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
 

    # Salvar imagem destacando a área do fundo encontrada, se necessário
    img_fundo_destacado = img.copy()
    cv2.drawContours(img_fundo_destacado, contours, -1, (0, 0, 255), 2)

    	
    # Calcula a área total dos contornos
    area_racao_funcao = sum(cv2.contourArea(contour) for contour in contours)

    return img_fundo_destacado, area_racao_funcao



def print_tela_red(imagem, texto, posicao):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    cv2.putText(imagem, texto, (50, posicao),font, 1,  
        (0, 0, 255),  
        2,cv2.LINE_4)
    return imagem

def print_tela_green(imagem, texto, posicao):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    cv2.putText(imagem, texto, (50, posicao),font, 1,  
        (0, 255, 0),  
        2,cv2.LINE_4)
    return imagem

def entrada():
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print('Com a câmera ligada digite a letra "c" para gravar a imagem do pote cheio e "t" para o pote de teste. Caso tenha as imagens aperte "p" para proseguir.')
    while True:
    
        
        
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
            
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k == ord('c'):
            cv2.imwrite('../img_projeto/pote-cheio.jpg', frame)
            print("Salvando imagem do pote vazio")
        
        if k == ord('t'):
            cv2.imwrite('../img_projeto/pote-teste.jpg', frame)
            print("Salvando imagem do pote para teste")
            
            
        if k == ord('p'):
            print("Iniciando programa")
            break
            

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()

selecao = input("Selecione o modo de uso [(a) - Live | (b) - Static]: ")
#modo ao vivo
if selecao == "a":
    area_racao = pote_cheio()
    if area_racao != 5:
        webcam()

#modo input
elif selecao == "b":
    escolha = input("(a) - Configurar agora | (b) - Usar default: ")
    if (escolha == "b"):
        imagem_pote_cheio = '../img_projeto/pote-azul-cheio.jpeg'
        imagem_pote = '../img_projeto/pote-azul-70.jpeg'
    elif (escolha == "a"):
        entrada()
        imagem_pote_cheio = '../img_projeto/pote-cheio.jpg'
        imagem_pote = '../img_projeto/pote-teste.jpg'
    B_A_img_cheio, area_racao = detectar_racao(imagem_pote_cheio, cor_fundo_hsv, True)
    B_A_img_teste, area_atual = detectar_racao(imagem_pote, cor_fundo_hsv, True)
    show_percent, status_pote =  calcula_percent(area_atual)
    texto_video = str(show_percent)
    if show_percent > 1:
        B_A_img_teste = print_tela_green(B_A_img_teste, status_pote, 20)

    elif show_percent > 0.5:
        B_A_img_teste = print_tela_green(B_A_img_teste, texto_video, 50)

        B_A_img_teste = print_tela_green(B_A_img_teste, status_pote, 20)

    else:
        B_A_img_teste = print_tela_red(B_A_img_teste, texto_video, 50)

        B_A_img_teste = print_tela_red(B_A_img_teste, status_pote, 20)

    # Display the resulting frame
    cv2.imshow('img_destacada_cheio', B_A_img_cheio)
    cv2.waitKey(0)
    cv2.imshow('img_destacada', B_A_img_teste)
    cv2.waitKey(0)
    