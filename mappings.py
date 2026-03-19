import numpy as np
bls_secpar1 = [29,9.45,8.2]
bls_secpar2 = [19.4,10.5,8.2]
bls_secpar3 = [21.4,9.5,8.2]
bls_params = np.array([bls_secpar2,bls_secpar3])

def pwlm(x,m1=0.8,m2=5,b1=4):
    a = b1/m1
    b2 = b1*(m2/m1)
    if x<=-a: return m1*x + b1 
    elif x>-a and x<0: return m2*x + b2
    elif x>=0 and x<a: return m2*x - b2
    elif x>=a: return m1*x - b1 

def der_pwl(x,m1=0.8,m2=5,b1=4):
    a = b1/m1 
    if x<=-a: return m1
    elif x>-a and x<0: return m2
    elif x>=0 and x<a: return m2
    elif x>=a: return m1

def bls_map(x,m,p,r):
    b = m*p
    if x>=-r and x <= 0:
        return np.mod(m*x+b,r)
    elif x>0 and x<=r:
        return -np.mod(-m*x+b,r)
    
def mapping_blsOrbit(xinit,m=bls_secpar1[0],p=bls_secpar1[1],r=bls_secpar1[2],orbit_size=0):
    x = np.linspace(-r,r,10000)
    y = []
    for i in range(len(x)):
        y.append(bls_map(x[i],m,p,r))

    if orbit_size>0:
        px, py = np.empty((2,orbit_size+1))
        px[0], py[0] = xinit, 0

        # Cobweb diagram

        for n in range(1, orbit_size, 2):
            px[n] = px[n-1]
            py[n] = bls_map(px[n-1],m,p,r)
            px[n+1] = py[n]
            py[n+1] = py[n]
        
        return x,y,px,py
        
    return x,y

def bls_composed(x,paramsSet=bls_params):
    gox = bls_map(x,paramsSet[0,0],paramsSet[0,1],paramsSet[0,2])
    hox = bls_map(gox,paramsSet[1,0],paramsSet[1,1],paramsSet[1,2])
    return abs(hox)

def mapping_blsCompOrbit(xinit,paramsSet=bls_params,orbit_size=0):
    x = np.linspace(-paramsSet[0,2],paramsSet[0,2],10000)
    y = []
    for i in range(len(x)):
        y.append(bls_composed(x[i]))

    if orbit_size>0:
        px, py = np.empty((2,orbit_size+1))
        px[0], py[0] = xinit, 0

        # Cobweb diagram

        for n in range(1, orbit_size, 2):
            px[n] = px[n-1]
            py[n] = bls_composed(px[n-1])
            px[n+1] = py[n]
            py[n+1] = py[n]
        
        return x,y,px,py
        
    return x,y

def bls_cutoffs(m,p,r):
    ncuts = 40
    xceros = np.zeros(ncuts)
    kas = np.zeros(ncuts)
    for i in range(ncuts):
        xceros[i] = (i*r)/m - p
        kas[i]=i
 
    return abs(xceros[(xceros>-r) & (xceros<0)])#,kas[(xceros>-r) & (xceros<0)]

def binary_split(xi,xsep):
    xsep = sorted(xsep,reverse=True)
    xbin=0
    print(xi,xsep,xbin)
    for i,xcut in enumerate(xsep):
        # print(xbin)
        if xi >= xcut:
            # print("ok")
            return xbin
        xbin+=(-1)**(i)

    return xbin

def frac_part(x):
    return x - np.floor(x)

def pwlm_cutoff(x,m1=0.8,m2=5): # Confined to family of maps PWLM:[-1,1]->[-1,1]
    a = 1/m2
    b1 = m1/m2
    if x<=-a: return m1*x + b1 
    elif x>-a and x<0: return m2*x + 1
    elif x>=0 and x<a: return m2*x - 1
    elif x>=a: return m1*x - b1 

def der_pwlmcutoff(x,m1=0.8,m2=5): 
    a = 1/m2
    if x<=-a: return m1
    elif x>-a and x<0: return m2
    elif x>=0 and x<a: return m2
    elif x>=a: return m1

def logistic(x,r=3.57):
    return r * x * (1 - x)

def der_logistic(x,r=3.57):
    return r-2*r*x

def tent(x,a=2):
    if x <= 0.5 and x >= 0: return a*x
    if x > 0.5 and x <= 1: return a*(1-x)

def der_tent(x,a=2):
    if x <= 0.5 and x >= 0: return a
    if x > 0.5 and x <= 1: return -a

def mapping_pwlmOrbit(xinit,dyn_sysparams=[0.8,5,40.8],orbit_size=0):
    b2 = (dyn_sysparams[1]/dyn_sysparams[0])*dyn_sysparams[2]
    x = np.linspace(-b2,b2,500)
    y = []
    for i in range(len(x)):
        y.append(pwlm(x[i],dyn_sysparams[0],dyn_sysparams[1],dyn_sysparams[2]))

    if orbit_size>0:
        px, py = np.empty((2,orbit_size))
        px[0], py[0] = xinit, 0

        # Cobweb diagram

        for n in range(1, orbit_size, 2):
            px[n] = px[n-1]
            py[n] = pwlm(px[n-1],dyn_sysparams[0],dyn_sysparams[1],dyn_sysparams[2])
            px[n+1] = py[n]
            py[n+1] = py[n]
        
        return x,y,px,py
        
    return x,y

def calc_bif_lyap_pwl(x0,xi,xf,n,arg,fixedparams):
    # Calculate 
    n_iter = 500
    m_range = np.linspace(xi,xf,n) 
    M = [] # Array of M2 values, x axis for bifurcation
    X = []
    lyapunov = []
    
    for m in m_range:
        x = x0 
        ly = 0
        for i in range(n+n_iter):
            match arg:
                case "1":
                    ly += np.log(abs(der_pwl(x,m1=m,m2=fixedparams[1],b1=fixedparams[2]))) 
                    x = pwlm(x,m1=m,m2=fixedparams[1],b1=fixedparams[2])
                case '2':    
                    ly += np.log(abs(der_pwl(x,m1=fixedparams[0],m2=m,b1=fixedparams[2]))) 
                    x = pwlm(x,m1=fixedparams[0],m2=m,b1=fixedparams[2])
                case 'b':
                    ly += np.log(abs(der_pwl(x,m1=fixedparams[0],m2=fixedparams[1],b1=m))) 
                    x = pwlm(x,m1=fixedparams[0],m2=fixedparams[1],b1=m)
                case _:
                    print("Select argument of pwl.")
                    return

            if i >= n:
                M.append(m)
                X.append(x)
                
        lyapunov.append(ly/(n+n_iter)) 
        
    return M,X,m_range,lyapunov

def calc_bif_tent(x0,xi,xf,n):
    # Calculate 
    n_iter = 500
    m_range = np.linspace(xi,xf,n) 
    M = [] # Array of M2 values, x axis for bifurcation
    X = []
    lyapunov = []
    
    for m in m_range:
        x = x0 
        ly = 0
        for i in range(n+n_iter):
            ly += np.log(abs(der_tent(x,a=m))) 
            x = tent(x,a=m)

            if i >= n:
                M.append(m)
                X.append(x)
                
        lyapunov.append(ly/(n+n_iter)) 
        
    return M,X,m_range,lyapunov

def calc_bif_logistic(x0,xi,xf,n):
    # Calculate 
    n_iter = 500
    m_range = np.linspace(xi,xf,n) 
    M = [] # Array of M2 values, x axis for bifurcation
    X = []
    lyapunov = []
    
    for m in m_range:
        x = x0 
        ly = 0
        for i in range(n+n_iter):
            ly += np.log(abs(der_logistic(r=m,x=x))) 
            x = logistic(r=m,x=x)

            if i >= n:
                M.append(m)
                X.append(x)
                
        lyapunov.append(ly/(n+n_iter)) 
        
    return M,X,m_range,lyapunov
                
def orbit_mix_tseries(x01,x02,delay,f1,f2):
    x1 = [x01]
    x2 = [x02]
    m1 = []
    m2 = []
    m1_seq = 0
    m2_seq = 0
    delayhf = delay/2
   
    for i in range(6000):
        x1.append(pwlm(x1[i],f1[0],f1[1],f1[2])) #x1n+1=f(x1n)
        x2.append(pwlm(x2[i],f2[0],f2[1],f2[2]))

        if i>= delay:
            m1_seq = (x1[int(i-(delay))]+x1[int(i-(delayhf))]+x1[int(i)])%256
            m2_seq = (x2[int(i-(delay))]+x2[int(i-(delayhf+1))]+x2[int(i)])%256
            m1.append(m1_seq)
            m2.append(m2_seq)

    return(m1,m2)

def calc_lyapgrad_pwl(x0,n,fixedparams,m1range=[0.01,1],m2range=[0.1,20]):
    # Calculate 
    n_iter = 300
    mranges=np.array([m1range,m2range])
    m1_values = np.linspace(mranges[0,0],mranges[0,1],n)
    m2_values = np.linspace(mranges[1,0],mranges[1,1],n)
    ms_values = [m1_values,m2_values]
    lyapgrad = []
    
    for m2 in m2_values:
        lyapunov = []
        for m1 in m1_values:
            x = x0 
            ly = 0
            for i in range(n+n_iter):
                ly += np.log(abs(der_pwl(x,m1=m1,m2=m2,b1=fixedparams[2]))) 
                x = pwlm(x,m1=m1,m2=m2,b1=fixedparams[2])
        
            lyapunov.append(ly/(n+n_iter)) 
        lyapgrad.append(lyapunov)

        
    return ms_values,lyapgrad

def calc_lyapgrad_pwlmcutoff(x0,n,m1range=[0.01,1],m2range=[0.1,20]):
    # Calculate 
    n_iter = 300
    mranges=np.array([m1range,m2range])
    m1_values = np.linspace(mranges[0,0],mranges[0,1],n)
    m2_values = np.linspace(mranges[1,0],mranges[1,1],n)
    ms_values = [m1_values,m2_values]
    lyapgrad = []
    
    for m2 in m2_values:
        lyapunov = []
        for m1 in m1_values:
            x = x0 
            ly = 0
            for i in range(n+n_iter):
                ly += np.log(abs(der_pwl(x,m1=m1,m2=m2))) 
                x = pwlm(x,m1=m1,m2=m2)
        
            lyapunov.append(ly/(n+n_iter)) 
        lyapgrad.append(lyapunov)

        
    return ms_values,lyapgrad
