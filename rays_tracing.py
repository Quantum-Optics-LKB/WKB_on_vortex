#%%
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import RK45
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
from azim_avg import mean_azim
from functools import partial
import os

print(os.getcwd())
#%%
corr = 0.35
F = 1.3

folder = "/run/user/1000/gvfs/sftp:host=patriot.lkb.upmc.fr,user=kguerrero/partages/EQ15B/LEON-15B/DATA/Polaritons/Rotating_Geometry/simu/noisy_vortex_for_disp/data_set_noise=0_dx=1.0_dt=0.5"

#F = np.load(folder+"/F_ph_v_steady_state%s.npy"%set_name)
# fig, ax = plt.subplots(1,2)
# ax[0].imshow(np.angle(F), extent=[-128,128,128,-128])
# ax[1].imshow(np.abs(F), extent=[-128,128,128,-128])
# plt.show()

# pol_steady = np.load(folder+"/pol_mean_field%s.npy"%set_name)
# fig, ax = plt.subplots(1,2)
# ax[0].imshow(np.angle(pol_steady), extent=[-128,128,128,-128])
# ax[1].imshow(np.abs(pol_steady), extent=[-128,128,128,-128])
# plt.show()

m_lp=0.32
h=0.654
D = np.load(folder+"/eff_detuning.npy")
D = mean_azim(D)[:80]
gn = np.load(folder+"/gn.npy")
gn = mean_azim(gn)[:80]
c = np.sqrt(h*gn/m_lp)
vr = np.load(folder+"/vr.npy")
vr = mean_azim(vr)[:80]
vazim = np.load(folder+"/vazim.npy")
vazim = mean_azim(vazim)[:80]
v = np.sqrt(vazim**2+vr**2)

def disp(m=10, r=40, N=65):
    r = int(r)
    P = np.linspace(-1, 1, N)
    W_ev_p = []
    W_ev_m = []
    for k in range(N):
        disp_fld_homo = np.sqrt(h**2*(P[k]**2+(m/r)**2)**2/4/m_lp**2+h*(P[k]**2+(m/r)**2)*(2*gn[r]-D[r])/m_lp+(gn[r]-D[r])*(3*gn[r]-D[r]))
        kr = (P[k]*vr[r]+m*vazim[r]/r)
        w_ev_p = np.real(kr + disp_fld_homo)
        w_ev_m = np.real(kr - disp_fld_homo)
        W_ev_p.append(w_ev_p)
        W_ev_m.append(w_ev_m)
    return P, W_ev_p, W_ev_m
pp, w_ev_p, w_ev_m = disp()

def omega_plus(m, r, p):
    disp_fld_homo = np.sqrt(h**2*(p**2+(m/r)**2)**2/4/m_lp**2+h*(p**2+(m/r)**2)*(2*gn[r]-D[r])/m_lp+(gn[r]-D[r])*(3*gn[r]-D[r]))
    kr = (p*vr[r]+m*vazim[r]/r)
    return np.real(disp_fld_homo + kr)

def omega_minus(m, r, p):
    disp_fld_homo = np.sqrt(h**2*(p**2+(m/r)**2)**2/4/m_lp**2+h*(p**2+(m/r)**2)*(2*gn[r]-D[r])/m_lp+(gn[r]-D[r])*(3*gn[r]-D[r]))
    kr = (p*vr[r]+m*vazim[r]/r)
    return np.real(kr-disp_fld_homo)

def idx(arr, value):
    return np.argmin(np.abs(arr-value))
#%%
r_min = 5
r_max = 65
p_min = -1
p_max = 1
N = r_max-r_min+1
r_grid = np.linspace(r_min, r_max, N)
p_grid = np.linspace(p_min, p_max, N)
H_values = np.zeros((N, N))
H_interp_values = np.zeros((N, N))
Wplus_values = np.zeros((N, N))
Wminus_values = np.zeros((N, N))

#Example for harmonic oscillator
# r_min = 10
# r_max = 65
# p_min = -1
# p_max = 1
# r_grid = np.linspace(r_min, r_max, r_max-r_min+1)
# p_grid = np.linspace(p_min, p_max, r_max-r_min+1)
# R, P = np.meshgrid(r_grid, p_grid)
# H_values = R**2 / 2 + P**2 / 2  # Example Hamiltonian for demonstration

#%%
def numerical_partial_derivative(func, x, idx, delta=1e-5):
    x1 = np.copy(x)
    x2 = np.copy(x)
    x1[idx] += delta
    x2[idx] -= delta
    derivative = (func(x1) - func(x2)) / (2 * delta)
    return derivative.item() if derivative.size == 1 else derivative  # Use item() for scalar

def partial_derivative_r(H_interp, r, p, delta=1e-5):
    return numerical_partial_derivative(lambda rp: H_interp(rp), np.array([r, p]), 0, delta)

def partial_derivative_p(H_interp, r, p, delta=1e-5):
    return numerical_partial_derivative(lambda rp: H_interp(rp), np.array([r, p]), 1, delta)

def hamiltonian_system(t, y, H_interp, delta=1e-5):
    r, p = y
    dr_dt = partial_derivative_p(H_interp, r, p, delta)
    dp_dt = -partial_derivative_r(H_interp, r, p, delta)
    return [dr_dt, dp_dt]
#%%
# Initial conditions
m = -10
for r in r_grid:
    for p in p_grid:
        wplus = omega_plus(m, int(r), p)
        wminus = omega_minus(m, int(r), p)
        Wplus_values[idx(r_grid, r), idx(p_grid, p)] = wplus
        Wminus_values[idx(r_grid, r), idx(p_grid, p)] = wminus

Wplus_interpolator = RegularGridInterpolator((r_grid, p_grid), Wplus_values[::1, ::1], bounds_error=False, fill_value=None)
Wminus_interpolator = RegularGridInterpolator((r_grid, p_grid), Wminus_values[::1, ::1], bounds_error=False, fill_value=None)

#sound ring
# r0 = 50.8251
# p0 = -0.00976
#folder = os.getcwd() + "/sound_ring/"

#inside trapped mode
#r0 = 40
#p0 = +0.3388
#folder = os.getcwd() + "/trapped_inside_mode/"

#outside mode
# r0 = 64.9
# p0=-0.2088
# folder = os.getcwd() + "/outside_mode/"


theta0 = 0
theta0_dot = -0.01*(2*np.pi) # Initial angular velocity rad/s
y0 = [r0, p0]
w = Wplus_interpolator(y0)

H_values = -0.5*(w-Wminus_values)*(w-Wplus_values)
H_interpolator = RegularGridInterpolator((r_grid, p_grid), H_values[::1, ::1], bounds_error=False, fill_value=None)

for r in r_grid:
    for p in p_grid:    
        H_interp_values[idx(r_grid, r), idx(p_grid, p)] = H_interpolator([r, p])

thinner_r_grid = np.linspace(r_min, r_max, 5000)
thinner_p_grid = np.linspace(p_min, p_max, 5000)
R, P = np.meshgrid(thinner_r_grid, thinner_p_grid)
H_thin_values = H_interpolator((R, P))

plt.figure("H")
extent = [r_min, r_max, p_max, p_min]
plt.imshow(H_thin_values>0, extent=extent, aspect='auto')
plt.scatter(r0, p0, c='r', marker='x', label='Initial Position')
plt.colorbar()
plt.show()
#%%
# Time span
t0 = 0.0
t1 = 400
dt = 0.01

# Lists to store results
t_values = []
y_values = []
E_values = []

# Partial function to include parameters
ode_system = partial(hamiltonian_system, H_interp=H_interpolator, delta=1e-5)

# RK45 solver
solver = RK45(ode_system, t0, y0, t1, max_step=dt)

while solver.status == 'running':
    solver.step()
    t_values.append(solver.t)
    y_values.append(solver.y)
    E_values.append(H_interpolator(solver.y))
    print(solver.t)
    #if solver.t < dt or (t1 - solver.t)%5 < dt:
    # if True:
    #     print(solver.y)
    #     print(H_interpolator(solver.y))
    #     print(H_values[idx(r_grid, solver.y[0]), idx(p_grid, solver.y[1])])

theta = theta0 + theta0_dot * np.array(t_values)

# Convert to arrays
t_values = np.array(t_values)
y_values = np.array(y_values)
E_values = np.array(E_values)
#%%
#Plot radius vs time

plt.rcParams.update({
    'axes.titlesize': 20,     # Title size
    'axes.labelsize': 20,     # X and Y labels size
    'xtick.labelsize': 20,    # X tick labels size
    'ytick.labelsize': 20,    # Y tick labels size
    'legend.fontsize': 20     # Legend font size
})

# Create figure and subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

# Plot radius vs time
axs[0].plot(t_values, y_values[:, 0], color='k')
axs[0].set_xlabel('t(ps)')
axs[0].set_ylabel('r(µm)')
axs[0].set_ylim(r_min, r_max)
axs[0].set_title('Radius vs Time')
axs[0].grid(True)

# Plot momentum vs time
axs[1].plot(t_values, y_values[:, 1], color='k')
axs[1].set_xlabel('t(ps)')
axs[1].set_ylabel('p(µm^-1)')
axs[1].set_ylim(p_min, p_max)
axs[1].set_title('Radial Momentum vs Time')
axs[1].grid(True)

# Plot phase space
axs[2].scatter(y_values[:, 1], y_values[:, 0], color='k')
axs[2].set_xlabel('p(µm^-1)')
axs[2].set_ylabel('r(µm)')
axs[2].set_ylim(r_min, r_max)
axs[2].set_xlim(p_min, p_max)
axs[2].set_title('Phase Space Plot')
axs[2].grid(True)

plt.tight_layout()
plt.savefig(folder+'phase_space_trajectories.svg')
plt.savefig(folder+'phase_space_trajectories.png')
plt.show()

# Plot momentum vs time
plt.figure(figsize=(18, 10))
plt.plot(t_values, y_values[:, 1], color='k')
plt.xlabel('t(ps)')
plt.ylabel('p(µm^-1)')
plt.title('Radial Momentum vs Time')
plt.grid(True)
plt.savefig(folder+'radial_momentum_vs_time.svg')
plt.savefig(folder+'radial_momentum_vs_time.png')
plt.show()

# Plot radius vs time
plt.figure(figsize=(18, 10))
plt.plot(t_values, y_values[:, 0], color='k')
plt.xlabel('t(ps)')
plt.ylabel('r(µm)')
plt.title('Radius vs Time')
plt.grid(True)
plt.savefig(folder+'radius_vs_time.svg')
plt.savefig(folder+'radius_vs_time.png')
plt.show()

#Plot phase space from H=0
plt.figure(figsize=(18, 10))
#e_val = np.mean(E_values)
e_val = 0
print(e_val)
H_zeros = H_thin_values < e_val
extent = [r_min, r_max, p_max, p_min]
plt.imshow(H_zeros, extent=extent, aspect='auto', cmap='gray_r',vmax=2)
plt.plot(y_values[:, 0], y_values[:, 1], label='Trajectory', color='b', linewidth=5)
plt.scatter(r0, p0, c='r', label='Initial Position', cmap='inferno', s=100, marker='o')
plt.title('H=0 vs Trajectories')
plt.xlabel('r(µm)')
plt.ylabel('p(µm^-1)')
plt.legend()
plt.savefig(folder+'H=0_vs_trajectories.svg')
plt.savefig(folder+'H=0_vs_trajectories.png')
plt.show()

#Trajectory in real space
x = y_values[:, 0] * np.cos(theta)
y = y_values[:, 0] * np.sin(theta)
x0 = r0 * np.cos(theta0)
y0 = r0 * np.sin(theta0)
v0_pol = np.array([p0, r0*theta0_dot])
v0y = np.array([v0_pol[0] * np.sin(theta0) + v0_pol[1] * np.cos(theta0)])
v0x = np.array([v0_pol[0] * np.cos(theta0) - v0_pol[1] * np.sin(theta0)])
v0 = np.array([v0x, v0y])

plt.figure(figsize=(16, 10))
#plot circle at r_max and r_min
plt.plot(r_max*np.cos(np.linspace(0, 2*np.pi, 100)), r_max*np.sin(np.linspace(0, 2*np.pi, 100)), 'k', linestyle='--')
plt.plot(r_min*np.cos(np.linspace(0, 2*np.pi, 100)), r_min*np.sin(np.linspace(0, 2*np.pi, 100)), 'k', linestyle='--')
#plot rays
plt.scatter(x, y, c=t_values, cmap='inferno', marker='.', linewidths=0, label='(x,y)(t)')
plt.colorbar(label='Time(ps)')
plt.scatter(x0, y0, c='r', marker='x', label='Initial Position', s=40)
#plt.quiver(x0, y0, v0[0], v0[1], color='r', label='Initial Velocity')
plt.xlabel('y')
plt.ylabel('x')
# plt.xlim(-r_max, r_max)
# plt.ylim(r_max, -r_max)
plt.title('sound rays')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.savefig(folder+'real_space_trajectories.svg')
plt.savefig(folder+'real_space.trajectories.png')
plt.show()

def scatter_x_y_t(x, y, t, x0, y0, v0, r_min, r_max, folder):
    plt.figure(figsize=(16, 10))
    #plot circle at r_max and r_min
    plt.plot(r_max*np.cos(np.linspace(0, 2*np.pi, 100)), r_max*np.sin(np.linspace(0, 2*np.pi, 100)), 'k', linestyle='--')
    plt.plot(r_min*np.cos(np.linspace(0, 2*np.pi, 100)), r_min*np.sin(np.linspace(0, 2*np.pi, 100)), 'k', linestyle='--')
    #plot rays
    plt.scatter(x, y, c='k', cmap='inferno', marker='.', linewidths=0, label='(x,y)(t)', s=200)
    #plt.colorbar(label='Time(ps)')
    plt.scatter(x0, y0, c='r', marker='x', label='Initial Position', s=50)
    #plt.quiver(x0, y0, v0[0], v0[1], color='r', label='Initial Velocity')
    plt.xlabel('y')
    plt.ylabel('x')
    # plt.xlim(-r_max, r_max)
    # plt.ylim(r_max, -r_max)
    plt.title('sound at t=%sps'%str(int(t)))
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    # plt.savefig(folder+'real_space_t_%s.svg'%str(t))
    plt.savefig(folder+'real_space_t=%strajectories.png'%str(int(t)))
    plt.show()
#%%
def make_mp4_anim(t_values=t_values):
    
    for i in tqdm(range(len(t_values))):
        if t_values[i]%0.1<0.01:
            scatter_x_y_t(x[i], y[i], t_values[i], x0, y0, v0, r_min, r_max, folder)
        
    img = cv2.imread(folder + 'real_space_t=%strajectories.png'%str(t_values[0]))
    frameSize = (img.shape[1], img.shape[0])

    out = cv2.VideoWriter(folder+'traj_anim.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 3, frameSize)
    
    for t in tqdm(t_values):
        if t%0.1<0.01:
            img = cv2.imread(folder + 'real_space_t=%strajectories.png'%str(t))
            out.write(img)
    
    out.release()
    print(out)
    
#make_mp4_anim(t_values=t_values)
#%%
