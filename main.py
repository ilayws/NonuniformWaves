import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
from scipy.ndimage import gaussian_filter
from time import sleep

# -----------------------------------

#Parameters
T = 700; simu_mult = 10
L = 5
curr_dx = 10
dx = 0.01; dt = 0.01
def c_func(x):
    out = np.ones(x.size)
    out[np.argwhere(x<=0.5*L)] = 0.5
    return out
# def c_func(x):
#     return 1 - 0.9*x/L

resm = 50

PULSE = True
pulse_f = 2
pulse_t = 1
halves = 2


# -----------------------------------

t = 0; N = int(L/dx)
M = np.zeros((N,3)) # String
C = dx/dt # Constant = dx/dt
x = np.linspace(0,L,N-2); c = c_func(x)

plt.xlabel("x")
plt.ylabel("c(x)")
plt.plot(x,c)
plt.show()

# -----------------------------------

f_list = []
t_list = []
a_list = []

# -----------------------------------

# Config matplot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(121)
#ax2 = fig.add_subplot(122)
ax3 = fig.add_subplot(122)

ax1.set_xlim(0,L); ax1.set_ylim(-2,2); ax1.set_xlabel("x(m)"); ax1.set_ylabel("y(m)");
#ax2.set_xlim(0,L); ax2.set_ylim(0,1.5); ax2.set_xlabel("x(m)"); ax2.set_ylabel("k(1/m)"); ax2.set_title("Instantaneous frequency")
ax3.set_xlim(0,25); ax3.set_ylim(0,1.5); ax3.set_xlabel("k(1/m)"); ax3.set_ylabel("y(m)"); ax3.set_title("Fourier Transform")

wave, = ax1.plot(np.arange(N)*dx,M[:,2], c="k")
#insta, = ax2.plot([],[],'o')
fourier, = ax3.plot([],[],'o')

# -----------------------------------

def update():
    M[1:-1,2] = 2*M[1:-1,1]-M[1:-1,0] + np.power(c/C,2)*( M[2:,1]+M[:-2,1]-2*M[1:-1,1] )
    M[:,0] = M[:,1]
    M[:,1] = M[:,2]

def display():
    ax1.set_title("Wave | t=" + str(np.round(t,2))+"s")
    wave.set_ydata(M[:,2])
    fig.canvas.draw()
    fig.canvas.flush_events()

# ------------------------------

def standing_wave(m, l, h):
    f = (h) / (2*L)
    W = np.sin(2*np.pi*f*dx*np.arange(m.shape[0]))
    m[:,[0,1]] = np.repeat(W,2).reshape(int(W.size),2)
    m[[0,-1],:] = 0

def pulse(f, time):
    M[0,2] = 0
    if (t < time):
        M[0,2] = np.sin(2*np.pi*f*t)

# ------------------------------

# Fourier
def get_spatfreq(m):
    f_vector = fft(m,m.shape[0]*resm)
    f_size = abs(f_vector)
    k = np.abs(fftfreq(m.shape[0]*resm,dx))

    der1 = np.diff(f_size) / np.diff(k)
    der2 = np.diff(der1) / np.diff(k[:-1])
    sign = np.sign(der1[1:]*der1[:-1])
    maxidx = np.argwhere((sign < 0) & (der2 < 0))
    i = np.argwhere(k[maxidx] == 0)
    maxidx = np.delete(maxidx, i)
    if len(maxidx) == 0:
        K = 0; A = 0
    else:
        fundidx = np.argmax(f_size[maxidx])
        K = k[maxidx][fundidx]*2*np.pi; A = f_size[fundidx]
    fourier.set_xdata(k)
    fourier.set_ydata(f_size/np.max(f_size))
    return K, A


# f = kc/2pi
def get_tempfreq(k, average=True, s=0, e=-1):
    expanded_c = np.concatenate(  (np.array(c[0]).reshape(1), c[:].reshape(N-2))  )
    if average:
        F = k*np.mean(expanded_c[s:e]) / (2*np.pi)
    else:
        F = k*expanded_c / (2*np.pi)
    return F

# Hilbert
def get_instafreq(m, dom="k"):
    analytic_signal = hilbert(m)
    A = np.abs(analytic_signal)
    pi = np.unwrap(np.angle(analytic_signal))
    fi = np.diff(pi)
    if dom == "k":
        fi /= dx
    elif dom == "t":
        fi /= (2*np.pi*dt)
    fi = gaussian_filter(fi, sigma=4)
    #insta.set_xdata(np.arange(dx,L,dx))
    #insta.set_ydata(fi/2/np.pi)
    #insta.set_ydata(get_tempfreq(M,sections,fi))
    return fi, A

def run():
    global t
    
    if not PULSE:
        standing_wave(M,L,2)
    for i in range(int((T/dt)/simu_mult)):
        display()
        for j in range(simu_mult):
            if PULSE:
                pulse(pulse_f,pulse_t)
            update(); t+=dt
            #k, A = get_instafreq(M[:,2], dom="k")
            #f = get_tempfreq(k, False)
            k, A = get_spatfreq(M[0:100,2])
            f = get_tempfreq(k, True, 0, 100)
            #f_list.append(f[-1])
            #t_list.append(t)
            #a_list.append(A)

run()
plt.ioff()
#plt.figure()
#plt.plot(t_list, f_list)
#plt.show()
