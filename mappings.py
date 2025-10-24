import math
import numpy as np
import matplotlib.pyplot as plt
import secrets as scrt

def pwlm(x,m1=0.8,m2=5,b1=40.8):
    a = b1/m1
    b2 = b1*(m2/m1)
    if x<=-a: return m1*x + b1 
    elif x>-a and x<0: return m2*x + b2
    elif x>=0 and x<a: return m2*x - b2
    elif x>=a: return m1*x - b1 

def der_pwl(x,m1=0.8,m2=5,b1=40.8):
    a = b1/m1 
    if x<=-a: return m1
    elif x>-a and x<0: return m2
    elif x>=0 and x<a: return m2
    elif x>=a: return m1

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
    if x <= 0.5 and x >= 0: return a*x
    if x > 0.5 and x <= 1: return -a

def mapping_pwlmOrbit(dyn_system,xinit,dyn_sysparams=[0.8,5,40.8],orbit_size=0):
    b2 = (dyn_sysparams[1]/dyn_sysparams[0])*dyn_sysparams[2]
    x = np.linspace(-b2,b2,500)
    y = []
    for i in range(len(x)):
        y.append(dyn_system(x[i],dyn_sysparams[0],dyn_sysparams[1],dyn_sysparams[2]))

    if orbit_size>0:
        px, py = np.empty((2,orbit_size+1))
        px[0], py[0] = xinit, 0

        # Cobweb diagram

        for n in range(1, orbit_size, 2):
            px[n] = px[n-1]
            py[n] = dyn_system(px[n-1],dyn_sysparams[0],dyn_sysparams[1],dyn_sysparams[2])
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
