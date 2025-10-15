import numpy as np

class Sbox:
    def __init__(self,sbarray):
        self.table=sbarray
        self.size=len(sbarray)
        self.nbits=int(np.log2(self.size))
        self.rot_cont=0

    def __decimal_to_binary_array(self, bit_length=8):
    #Convert each entry of the S-box into a binary representation
        return [np.array([int(x) for x in f"{val:0{bit_length}b}"]) for val in self.table]
    
    def __extract_boolean_functions(self,sbox_binary):
        #Extract boolean functions from the binary representation of the S-box
        n_bits = len(sbox_binary[0])
        boolean_functions = [[] for _ in range(n_bits)]
        for entry in sbox_binary:
            for i in range(n_bits):
                boolean_functions[i].append(entry[i])
        return boolean_functions
    
    def __extract_output_bits(self,bit_pos):
        return [(self.table[i] >> bit_pos) & 1 for i in range(self.size)]

    def __binary_inner_product(self,x,w):
    #binary dot product x · w (mod 2)
        return bin(x & w).count('1') % 2

    def __nonlinearity(self,binary_func):
        #Walsh transform
        W = np.zeros(self.size)
        for w in range(self.size):
            W[w] = sum((-1)**(binary_func[x] ^ self.__binary_inner_product(x, w)) for x in range(self.size))

        max_walsh = np.max(np.abs(W))
        return (2**(self.nbits-1)) - (max_walsh / 2)

    def nonlinearity_analysis(self):
        #Calculate the nonlinearity of the 8 boolean functions corresponding to the S-box
        sbox_binary = self.__decimal_to_binary_array(bit_length=self.nbits)
        boolean_functions = self.__extract_boolean_functions(sbox_binary)
        
        nonlinearities = []
        for f in boolean_functions:
            nl = self.__nonlinearity(f)
            nonlinearities.append(nl)
        return nonlinearities

    def sac_analysis(self):
        S=self.table
        n = int(np.log2(len(S)))       # Número de bits de entrada
        total_inputs = len(S)          # 2^n
        sac_matrix = np.zeros((n, n))  # Inicializar matriz de resultados
        
        for i in range(n):             # Por cada bit de entrada
            bit_mask = 1 << i          # Máscara para el bit i
            output_diffs = []          # Almacenar diferencias de salida
            
            for x in range(total_inputs):
                # Calcular diferencia al cambiar el bit i
                diff = S[x] ^ S[x ^ bit_mask]
                output_diffs.append(diff)
            
            # Contar cambios por bit de salida (vectorizado)
            for j in range(n):
                bit_j_mask = 1 << j
                count = sum(1 for diff in output_diffs if diff & bit_j_mask)
                sac_matrix[i, j] = count / total_inputs
        
        return np.round(sac_matrix, 4)
    
    def bicsac_analysis(self):
        n = self.nbits
        BIC_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                change_count = 0
                total_tests = 0

                for x in range(self.size):
                    y = self.table[x]
                    for k in range(n):
                        x_prime = x ^ (1 << k)
                        y_prime = self.table[x_prime]

                        y_i = (y >> i) & 1
                        y_prime_i = (y_prime >> i) & 1
                        y_j = (y >> j) & 1
                        y_prime_j = (y_prime >> j) & 1

                        if (y_i != y_prime_i) ^ (y_j != y_prime_j):
                            change_count += 1
                        
                        total_tests += 1

                BIC_matrix[i][j] =change_count / (total_tests)

        return BIC_matrix
    
    def bicnonl_analysis(self):
        bic_nonlinearity_matrix = np.zeros((self.nbits,self.nbits),dtype=int)
        combined_function=0
        for i in range(self.nbits):
            for j in range(self.nbits):
                if i == j:
                    bic_nonlinearity_matrix[i][j] = 0
                else:
                    output_bits_i = self.__extract_output_bits(i)
                    output_bits_j = self.__extract_output_bits(j)
                    combined_function = [output_bits_i[k] ^ output_bits_j[k] for k in range(self.size)]
                    
                    # bic_nonlinearity_matrix[i][j] = int(nonLinearity(combined_function))
                    bic_nonlinearity_matrix[i][j] = int(self.__nonlinearity(combined_function))
                    
        return bic_nonlinearity_matrix

    def bic_average(self,bic_data):
        diag = []
        for i in range(len(bic_data)):
            for j in range(i+1,len(bic_data)):
                diag.append(bic_data[i,j])

        return np.average(diag),diag
    
    def is_bijective(self):
        # Precalcular tabla de paridad (GF(2)) para [0, 2^n - 1]
        parity_table = [i.bit_count() % 2 for i in range(self.size)]
        
        # Evaluar todas las combinaciones no nulas a ∈ {1, 2, ..., 2^n - 1}
        for a in range(1, self.size):
            count = 0
            for x in range(self.size):
                y = self.table[x]
                # Calcular F_a(x) = a · y (producto punto en GF(2))
                count += parity_table[a & y]  # += 0 o 1
            if count != (self.size // 2):  # ¿wt(F_a) = 2^{n-1}?
                return count
        return True

    def linear_probability(self):
        # Inicializar MELP
        max_lp = 0

        # Iterar sobre todas las máscaras a y b distintas de cero
        for a in range(1, self.size):
            for b in range(1, self.size):
                # Calcular la suma de (-1)^(a*x + b*f(x)) para cada x en el dominio de la S-box
                sum_bias = sum(
                    (-1) ** (self.__binary_inner_product(a,x) ^ self.__binary_inner_product(b,self.table[x]))
                    for x in range(self.size)
                )
                # Calcular probabilidad lineal para a y b
                lp = (sum_bias / self.size) ** 2 
                # Actualizar el máximo si es necesario
                max_lp = max(max_lp, lp)

        return max_lp

    def differential_probability(self):
        # Crear una tabla diferencial para almacenar las frecuencias
        differential_table = np.zeros((self.size, self.size), dtype=int)

        # Calcular la tabla diferencial
        for x in range(self.size):
            for delta_x in range(1, self.size):  # delta_x != 0
                # Aplicar diferencia de entrada
                x_prime = x ^ delta_x
                # Diferencia de salida
                delta_y = self.table[x] ^ self.table[x_prime]
                # Incrementar el contador para (delta_x, delta_y)
                differential_table[delta_x][delta_y] += 1

        # Calcular la probabilidad diferencial máxima
        max_dp = np.max(differential_table) / self.size
        return max_dp#, differential_table

    def tex_table(self,table):
        tex_table = []
        for ren in table:
            tex_table.append("&".join(str(np.round(x,3)) for x in ren))
            
        tex_table = "\\".join(str(x) for x in tex_table)
            
        return(tex_table)

    def tex_sbtable(self):
        n = 16
        tex_sbox = []
        for r in range(n):
            row = self.table[r*n:(r+1)*n]
            
            tex_sbox.append('&'.join(str(i) for i in row))

        tex_sbox = "\\".join(str(x) for x in tex_sbox)
        return tex_sbox

    def encode_image(self,image):
        image_arr = np.array(image)
        rows, cols = np.shape(image_arr)
        for i in range(rows):
            for j in range(cols):
                pixel_value = image_arr[i, j]
                image_arr[i, j] = self.table[pixel_value]

        return image_arr
    
    def decode_image(self,image):
        image_arr = np.array(image)
        rows, cols = np.shape(image_arr)
        for i in range(rows):
            for j in range(cols):
                encoded_val = image_arr[i, j]
                image_arr[i, j] = self.table.index(encoded_val)

        return image_arr
    
    ### DO ZIGZAGO ROTATIOOOOON

    def rotation(self,dir=1,k=1):
        #Right 
        if dir: 
            self.table=self.table[-k:]+self.table[:-k]
            self.rot_cont+=1
        #Left 
        else: 
            self.table=self.table[k:]+self.table[:k]
            self.rot_cont-=1

    def reset_table(self):
        self.table=self.table[self.rot_cont:]+self.table[:self.rot_cont]

    def zigzag_transform(self):
        n = 16  # Tamaño de la matriz (16x16)
        # Generar secuencia de índices (i, j) en orden zigzag
        zigzag_path = []
        i, j = 0, 0
        up = True
        total = n * n
        
        for _ in range(total):
            zigzag_path.append((i, j))
            
            if up:  # Movimiento en diagonal hacia arriba-derecha
                if i > 0 and j < n - 1:
                    i -= 1
                    j += 1
                else:
                    up = False
                    if j < n - 1:
                        j += 1
                    else:
                        i += 1
            else:  # Movimiento en diagonal hacia abajo-izquierda
                if j > 0 and i < n - 1:
                    i += 1
                    j -= 1
                else:
                    up = True
                    if i < n - 1:
                        i += 1
                    else:
                        j += 1
        
        # Construir nueva S-box usando el orden zigzag
        
        new_sbox = [self.table[i * n + j] for (i, j) in zigzag_path]
        self.table = new_sbox
        return 
        

