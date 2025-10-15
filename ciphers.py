import math
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sbox import Sbox
import time

def pwlm(x,m1=0.8,m2=5,b1=40.8):
    a = b1/m1
    b2 = b1*(m2/m1)
    if x<=-a: return m1*x + b1 
    elif x>-a and x<0: return m2*x + b2
    elif x>=0 and x<a: return m2*x - b2
    elif x>=a: return m1*x - b1 

def generate_sbox(x01,x02,delay=10,f1=[0.8,5,40.8],f2=[0.9,4,31]):
    x1 = [x01]
    x2 = [x02]
    m1_seq = 0
    m2_seq = 0
    delayhf = int(delay*0.5)
    sb = []

    i = 0
    while len(sb) < 256:
        x1.append(pwlm(x1[i],f1[0],f1[1],f1[2])) #x1(n+1)=f(x1(n))
        x2.append(pwlm(x2[i],f2[0],f2[1],f2[2]))

        if i>= delay:
            m1_seq = (x1[int(i-(delay))]+x1[int(i-(delayhf))]+x1[int(i)])%256
            m2_seq = (x2[int(i-(delay))]+x2[int(i-(delayhf+1))]+x2[int(i)])%256
            Zi = math.floor((m1_seq+m2_seq)%256)
            if Zi not in sb: sb.append(Zi)

        i+=1
            
    return sb

def alberti_cipher(image):
    arr = np.array(image)
    n = arr.shape[0]
    m = arr.shape[1]
    
    encode_arr = np.zeros((n,m))

    for i in range(n):
        x01,x02 = random.randint(0,255),random.randint(0,255)
        m1_params = np.random.random_integers(8000,10000,2)/10000
        m2_params = np.random.random_integers(2000,10000,2)/1000
        b1_params = np.random.random_integers(8000,50000,2)/1000
        func1 = [m1_params[0],m2_params[0],b1_params[0]]
        func2 = [m1_params[1],m2_params[1],b1_params[1]]
        sb = generate_sbox(x01,x02,10,func1,func2)
        for j in range(m):    
            encode_arr[i,j] = sb[arr[i,j]]
        
    return encode_arr

def encode_image(image, sbox):
    # Primera ronda: dirección directa (de izquierda a derecha, de arriba hacia abajo)
    rows, cols = np.shape(image)
    for i in range(rows):
        for j in range(cols):
            pixel_value = image[i, j]
            image[i, j] = sbox[pixel_value]

    return image

def plot_image_and_histogram(image_array):
    # Calcular el histograma
    histogram, bins = np.histogram(image_array, bins=256, range=(0, 256))

    # Crear la figura y los ejes
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))


    cm = plt.cm.get_cmap('Greys')
    # Mostrar la imagen en blanco y negro
    axs[0].imshow(image_array, cmap='gray', aspect='auto')
    axs[0].axis('off')
    #axs[0].set_title("Imagen en blanco y negro")

    # Mostrar el histograma
    axs[1].bar(bins[:-1], histogram, width=1, color='black', edgecolor='black', align='edge')
    axs[1].set_xlim(0, 255)
    #axs[1].set_title("Histograma")
    #axs[1].set_xlabel("Intensidad de Gris")
    axs[1].set_ylabel("Frecuencia",fontsize=18)


    #Crear la barra de escala de grises debajo del histograma
    x_span = bins.max()-bins.min()
    C = [cm(((x-bins.min())/x_span)) for x in bins]
    axs[1].bar(bins[:-1], histogram, width=bins[1]-bins[0], color=C, alpha=0.8)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return bins,histogram

def plot_histogram(image):

    image_array = np.array(image)

    # Calcular el histograma
    histogram, bins = np.histogram(image_array, bins=256, range=(0, 256))

    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot()

    ax.bar(bins[:-1], histogram, width=1, color='black', edgecolor='black', align='edge')
    ax.set_xlim(0, 255)
    #axs[1].set_title("Histograma")
    #axs[1].set_xlabel("Intensidad de Gris")
    
    ax.set_ylabel("Frecuencia",fontsize=17)


    #Crear la barra de escala de grises debajo del histograma
    x_span = bins.max()-bins.min()
    cm = plt.cm.get_cmap('Greys')
    C = [cm(((x-bins.min())/x_span)) for x in bins]
    ax.bar(bins[:-1], histogram, width=bins[1]-bins[0], color=C, alpha=0.8)


    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()

    return fig,ax

def plot_image(image):
    image_array = np.array(image)

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.imshow(image_array, cmap='gray')
    ax.axis('off')

    plt.tight_layout()
    plt.show()

def calculate_pixel_correlation(image):
   
    image_array = np.array(image, dtype=np.float64)
    
    # Correlación horizontal
    corr_horz = pearsonr(
        image_array[:, :-1].flatten(),  # Todos los píxeles excepto el último de cada fila
        image_array[:, 1:].flatten()   # Todos los píxeles excepto el primero de cada fila
    )[0]
    
    # Correlación vertical
    corr_vert = pearsonr(
        image_array[:-1, :].flatten(),  # Todos los píxeles excepto el último de cada columna
        image_array[1:, :].flatten()   # Todos los píxeles excepto el primero de cada columna
    )[0]
    
    # Correlación diagonal
    corr_diag = pearsonr(
        image_array[:-1, :-1].flatten(),  # Todos los píxeles excepto los de la última fila y columna
        image_array[1:, 1:].flatten()    # Todos los píxeles excepto los de la primera fila y columna
    )[0]
    
    return {
        "Horizontal": np.round(corr_horz,5),
        "Vertical": np.round(corr_vert,5),
        "Diagonal": np.round(corr_diag,5),
        "Avg": np.round((corr_diag+corr_horz+corr_vert)/3,5)
    }

def calculate_image_entropy(image):
    entropy=0
    img_arr = np.array(image)
    hist,_ = np.histogram(img_arr, bins=256, range=(0, 255), density=True)
    
    for i in range(len(hist)-1):    
        if hist[i]>0:
            entropy += hist[i]*np.log2(1/hist[i])
        else: continue

    
    return entropy

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	# err = np.sqrt(err)
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def psnr(imageA,imageB):
	mse_val = mse(imageA,imageB)
	psnr_val = 10*np.log10((255**2)/mse_val)#np.sqrt(mse_val))
	return mse_val,psnr_val

def enigmarot_cipher(image,xinits,params):
    k=3
    img_array = np.array(image)
    n,m = img_array.shape[0],img_array.shape[1]
    startT = time.time()
    sboxes = [Sbox(generate_sbox(xinits[i][0],xinits[i][1],f1=params[0],f2=params[1])) for i in range(k)]
    # print(sboxes[0].table)
    # print(sboxes[1].table)
    # print(sboxes[2].table)
    
    cipher_arr = np.zeros((n,m))

    for c in range(n):
        for r in range(m):
            aux = sboxes[0].table[int(img_array[c,r])]
            aux = sboxes[1].table[int(aux)]
            aux = sboxes[2].table[int(aux)]
            cipher_arr[c,r] = aux
            sboxes[k-k].rotation()
            # rot_cont[k-1]=sboxes[k-1].rot_cont
            if sboxes[k-k].rot_cont%256==0 and sboxes[k-k].rot_cont!=0:      
                sboxes[k-2].rotation()
                # rot_cont[k-2]+=1
                # print("Rot 2:")
                # print("table:"+str(sboxes[k-2].table))
                # print("rots:"+str(sboxes[k-2].rot_cont))
            
            if sboxes[k-2].rot_cont%256==0 and sboxes[k-2].rot_cont!=0:
                sboxes[k-1].rotation()
                # rot_cont[k-3]=sboxes[k-3].rot_cont
                sboxes[k-2].rot_cont=0 #?????
                # print("Rot 3:")
                # print("table:"+str(sboxes[k-3].table))
                # print("rots:"+str(sboxes[k-3].rot_cont))
                

            if sboxes[k-1].rot_cont%256==0 and sboxes[k-1].rot_cont!=0:
                print("ya es toda we ",sboxes[k-1].rot_cont)

    endT = time.time()
    print("Time:")
    print(endT - startT)
    print("End, rotcont:")
    print(sboxes[0].rot_cont,sboxes[1].rot_cont,sboxes[2].rot_cont)
    sboxes[0].reset_table()
    sboxes[1].reset_table()
    sboxes[2].reset_table()
    # print(sboxes[0].table)
    # print(sboxes[1].table)
    # print(sboxes[2].table)
    rot_cont = [sboxes[0].rot_cont,sboxes[1].rot_cont,sboxes[2].rot_cont]
    
    return cipher_arr,xinits,rot_cont,sboxes

def enigmarot_decipher(image,xinits,params):
    k=3
    img_array = np.array(image)
    n,m = img_array.shape[0],img_array.shape[1]
    # xinits = [[np.random.uniform(-255,255),np.random.uniform(-137.7,137.7)] for i in range(k)]
    startT = time.time()
    sboxes = [Sbox(generate_sbox(x[1][0],x[1][1],f1=params[0],f2=params[1])) for x in enumerate(xinits)]
    
    decipher_arr = np.zeros((n,m))

    for c in range(n):
        for r in range(m):
            aux = sboxes[2].table.index(img_array[c,r])
            aux = sboxes[1].table.index(aux)
            aux = sboxes[0].table.index(aux)
            decipher_arr[c,r] = aux
            sboxes[k-k].rotation()
            # rot_cont[k-1]=sboxes[k-1].rot_cont
            if sboxes[k-k].rot_cont%256==0 and sboxes[k-k].rot_cont!=0:      
                sboxes[k-2].rotation()
                # rot_cont[k-2]+=1
                # print("Rot 2:")
                # print(sboxes[k-2].table)
                # print(rot_cont)
            
            if sboxes[k-2].rot_cont%256==0 and sboxes[k-2].rot_cont!=0:
                sboxes[k-1].rotation()
                # rot_cont[k-3]=sboxes[k-3].rot_cont
                sboxes[k-2].rot_cont=0
                # print("Rot 3:")
                # print(sboxes[k-3].table)
                # print(rot_cont)
                

            if sboxes[k-1].rot_cont%256==0 and sboxes[k-1].rot_cont!=0:
                print("ya es toda we ",sboxes[k-1].rot_cont)
        
    endT = time.time()
    print("Time:")
    print(endT - startT)
    rot_cont = [sboxes[0].rot_cont,sboxes[1].rot_cont,sboxes[2].rot_cont]

    return decipher_arr,xinits,rot_cont,sboxes