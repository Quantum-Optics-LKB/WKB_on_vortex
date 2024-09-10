#%%
import os
import numpy as np
from azim_avg import mean_azim
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
matplotlib.rcParams['figure.figsize'] = [14, 10]
plt.rcParams['font.size'] = 30
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['legend.fontsize'] = 30
plt.rcParams['axes.titlepad'] = 30
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = False
plt.rcParams['grid.alpha'] = 0.5

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
N = 56

print(gn.shape)
print(vazim.shape)
print(vr.shape)
print(D.shape)

# plt.figure("spacetime")
# plt.plot(vazim, label = r"$v_{\theta}$")
# plt.plot(vr, label = "$v_{r}$")
# plt.plot(c, label = "$c_{sound}$")
# plt.grid(True, which='both')
# plt.ylabel("$[\mu m/ps]$")
# plt.xlabel("$r[\mu m]$")
# plt.legend()
# plt.savefig("velocities.pdf", dpi=600)

#plt.plot(vazim)
#plt.plot(vr)
# plt.figure("gn_vs_D")
# plt.plot(gn, label= "gn")
# plt.plot(D, label = "$\delta$")
# plt.grid(True, which='both')
# plt.ylabel("$[meV/\hbar]$")
# plt.xlabel("$r[\mu m]$")
# plt.ylim((-0.6,1))
# plt.legend()
# plt.savefig("gn_vs_delta.pdf", dpi=600)
# plt.show()

# pol = np.load(folder+"/pol.npy")[-10,:,:]
# extent = [-128,128,128,-128]



# gn = np.load(folder+"/gn.npy")
# c = np.sqrt(h*gn/m)
# vr = np.load(folder+"/vr.npy")
# vazim = np.load(folder+"/vazim.npy")
# v = np.sqrt(vazim**2+vr**2)

# fig, ax = plt.subplots(1,2)
# im0 = ax[0].imshow(c, cmap="gray", extent=extent)
# ax[0].set(title='cs',xlabel='x[µm]', ylabel='y[µm]')
# divider0 = make_axes_locatable(ax[0])
# cax0 = divider0.append_axes("right", size="5%", pad=0.05) 
# fig.colorbar(im0, cax=cax0)
# im1 = ax[1].imshow(np.angle(pol), extent=extent, cmap='twilight_shifted')
# ax[1].set(title='phase', xlabel='x[µm]')
# ax[1].set_yticks([])
# divider1 = make_axes_locatable(ax[1])
# cax1 = divider1.append_axes("right", size="5%", pad=0.05) 
# fig.colorbar(im1, cax=cax1)
# plt.savefig("den_phase.pdf", dpi = 600)
#%%

def disp(m=10, r=40):
    r = int(r)
    P = np.linspace(-1, 1, N)
    hW_ev_p = []
    hW_ev_m = []
    for k in range(N):
        disp_fld_homo = h*np.sqrt(h**2*(P[k]**2+(m/r)**2)**2/4/m_lp**2+h*(P[k]**2+(m/r)**2)*(2*gn[r]-D[r])/m_lp+(gn[r]-D[r])*(3*gn[r]-D[r]))
        hkr = h*(P[k]*vr[r]+m*vazim[r]/r)
        hw_ev_p = np.real(hkr + disp_fld_homo)
        hw_ev_m = np.real(hkr - disp_fld_homo)
        hW_ev_p.append(hw_ev_p)
        hW_ev_m.append(hw_ev_m)
    return P, hW_ev_p, hW_ev_m
pp, hw_ev_p, hw_ev_m = disp()

# r_min = 10
# r_max = 65
# N = r_max-r_min+1
# print(N)
# H_values = np.zeros((N, N))

# m = 10
# w = 0.356
# for r in np.linspace(r_min, r_max, N):
#     disp_result = disp(m=10, r=int(r))
#     w_ev_p = np.array(disp_result[1])[:]/h
#     w_ev_m = np.array(disp_result[2])[:]/h
#     H_values[:, int(r)-r_min] = -0.5*(w-w_ev_p)*(w-w_ev_m)

# r0 = 25
# m0 = -10
# wpump=0.14369402
# pp, hw_ev_p, hw_ev_m = disp(m=m0, r=r0)
# fig, ax = plt.subplots(1,1, figsize=(12,10))
# ax.set_title("$r=%s, m=%s$"%(str(r0), str(m0)))
# f, = plt.plot(pp, np.array(hw_ev_m)/h, color='red', label=r"$\hbar\omega_{Bog,-}(p)$")
# ff, = plt.plot(pp, np.array(hw_ev_p)/h, color='k', label=r"$\hbar\omega_{Bog,+}(p)$")
# ax.axhline(y=wpump, color='b', linestyle='dashed', label=r'$\omega_{scattering}$')
# plt.legend(loc='upper right')
# plt.ylabel(r"$\hbar\omega[meV]$")
# plt.xlabel(r"$p[µm^{-1}]$")
# plt.ylim(-0.5,1)
# plt.grid(True, which='both')
# axr = plt.axes([0.25, 0.05, 0.65, 0.03])
# axl = plt.axes([0.25, 0.10, 0.65, 0.03])
# plt.subplots_adjust(bottom=0.25)

# slider_r = Slider(axr, "r", 10, 79, valinit=r0)
# slider_l = Slider(axl, "m", -20, 20, valinit=m0)

# def update1(val):
#     f.set_ydata(np.array(disp(m=slider_l.val, r=slider_r.val)[1])/h)
#     ff.set_ydata(np.array(disp(m=slider_l.val, r=slider_r.val)[2])/h)
#     fig.canvas.draw_idle()

# slider_r.on_changed(update1)
# slider_l.on_changed(update1)
# plt.show()

#same thing but for m dispersions


def disp(p=0, r=40):
    # Define the range for m
    M = np.linspace(-M_max, M_max, N)
    hW_ev_p = []
    hW_ev_m = []
    
    # Loop over the range of m values, using M[k]
    for k in range(N):
        m = M[k] - C
        disp_fld_homo = h * np.sqrt(h**2 * (p**2 + (m/r)**2)**2 / (4 * m_lp**2) + h * (p**2 + (m/r)**2) * (2 * gn[int(r)] - D[int(r)]) / m_lp + (gn[int(r)] - D[int(r)]) * (3 * gn[int(r)] - D[int(r)]))
        hkr = h * (p * vr[int(r)] + m * vazim[int(r)] / r)
        hw_ev_p = np.real(hkr + disp_fld_homo)
        hw_ev_m = np.real(hkr - disp_fld_homo)
        hW_ev_p.append(hw_ev_p)
        hW_ev_m.append(hw_ev_m)
        
    return M, hW_ev_p, hW_ev_m

# Initial values for r, p, and m
r0 = 25
p0 = 0
m0 = 0
wpump = 0.14369402

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
ax.set_title(f"$r={r0}, p={p0}$")

C = 12
M_max = 30
ax.set_aspect(M_max)

M, hw_ev_p, hw_ev_m = disp(p=p0, r=r0)
f, = plt.plot(M, np.array(hw_ev_m) / h, color='red', label=r"$\hbar\omega_{Bog,-}$")
ff, = plt.plot(M, np.array(hw_ev_p) / h, color='k', label=r"$\hbar\omega_{Bog,+}$")
#ax.axhline(y=wpump, color='b', linestyle='dashed', label=r'$\omega_{scattering}$')
plt.legend(loc='upper right')
plt.ylabel(r"$\hbar\omega[meV]$")
plt.xlabel(r"$m+C$")
plt.ylim(-1, 1)
plt.grid(True, which='both')

# Add sliders for r and m
axr = plt.axes([0.25, 0.05, 0.65, 0.03])
axm = plt.axes([0.25, 0.10, 0.65, 0.03])
plt.subplots_adjust(bottom=0.25)

slider_r = Slider(axr, "r", 10, 79, valinit=r0, valstep=1)  # r slider with integer steps
slider_m = Slider(axm, "m", -20, 20, valinit=m0)

def update1(val):
    r = slider_r.val
    p = p0
    M, hw_ev_p, hw_ev_m = disp(p=p, r=r)  # Update the dispersions using current slider values
    f.set_ydata(np.array(hw_ev_m) / h)
    ff.set_ydata(np.array(hw_ev_p) / h)
    ax.set_title(f"$r={r}µm, p={p0}$")  # Update the title to reflect current r and p values
    fig.canvas.draw_idle()

slider_r.on_changed(update1)
slider_m.on_changed(update1)
plt.show()
