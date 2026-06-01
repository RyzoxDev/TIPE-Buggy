import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from itertools import product as itertools_product

# =========================
# PARAMÈTRES PHYSIQUES
# =========================
m     = 2.492
g     = 9.81
h     = 0.19
l     = 0.30
V_MAX = 50 / 3.6

# =========================
# CONFIGURATIONS DU BUGGY
# =========================
pneus = {
    "slick":  {"mu": {"sable": 0.28, "terre": 0.75, "beton": 0.85}, "a_moteur": 2.5},
    "sillon": {"mu": {"sable": 0.42, "terre": 0.84, "beton": 0.67}, "a_moteur": 2.5},
}
ressorts = {
    "souple": {"K": 388, "lam": 62.2},
    "rigide": {"K": 927, "lam": 96.1},
}
couleurs_sol = {"sable": "#d9c27f", "terre": "#8b5a2b", "beton": "#808080"}

def accel(pneu, sol):
    return min(pneus[pneu]["a_moteur"], pneus[pneu]["mu"][sol] * g)

# =========================
# CIRCUIT — Catmull-Rom fermé
# =========================
def catmull_rom_ferme(kp, n=50):   # n=50 au lieu de 60 : -17% de points
    def segment(p0, p1, p2, p3, alpha=0.5):
        def tj(ti, a, b):
            return ti + ((b[0]-a[0])**2 + (b[1]-a[1])**2)**alpha
        t0=0.; t1=tj(t0,p0,p1); t2=tj(t1,p1,p2); t3=tj(t2,p2,p3)
        pts = []
        for t in np.linspace(t1, t2, n, endpoint=False):
            A1=(t1-t)/(t1-t0)*np.array(p0)+(t-t0)/(t1-t0)*np.array(p1)
            A2=(t2-t)/(t2-t1)*np.array(p1)+(t-t1)/(t2-t1)*np.array(p2)
            A3=(t3-t)/(t3-t2)*np.array(p2)+(t-t2)/(t3-t2)*np.array(p3)
            B1=(t2-t)/(t2-t0)*A1+(t-t0)/(t2-t0)*A2
            B2=(t3-t)/(t3-t1)*A2+(t-t1)/(t3-t1)*A3
            pts.append((t2-t)/(t2-t1)*B1+(t-t1)/(t2-t1)*B2)
        return np.array(pts)
    N = len(kp)
    pts = np.vstack([segment(kp[(i-1)%N], kp[i], kp[(i+1)%N], kp[(i+2)%N]) for i in range(N)])
    return np.vstack([pts, pts[0]])

_kp_norm = np.array([
    [0.22, 0.55], [0.20, 0.72], [0.22, 0.80], [0.30, 0.82], [0.38, 0.78],
    [0.44, 0.84], [0.50, 0.76], [0.56, 0.84], [0.62, 0.76], [0.68, 0.82],
    [0.76, 0.76], [0.80, 0.68], [0.82, 0.60], [0.80, 0.52], [0.82, 0.44],
    [0.82, 0.36], [0.76, 0.28], [0.62, 0.22], [0.48, 0.20], [0.36, 0.22],
    [0.24, 0.26], [0.18, 0.36], [0.18, 0.46],
])

_pts_tmp   = catmull_rom_ferme(_kp_norm.tolist())
_S         = 500 / np.sqrt(np.diff(_pts_tmp[:,0])**2 + np.diff(_pts_tmp[:,1])**2).sum()
CIRCUIT    = catmull_rom_ferme((_kp_norm * _S).tolist())
N_PTS      = len(CIRCUIT)

_ZONES = [
    (0.00, 0.18, "beton"),
    (0.18, 0.42, "terre"),
    (0.42, 0.65, "sable"),
    (0.65, 0.85, "terre"),
    (0.85, 1.00, "beton"),
]
_ds_ref = np.sqrt(np.diff(CIRCUIT[:,0])**2 + np.diff(CIRCUIT[:,1])**2)
_frac   = np.concatenate([[0], np.cumsum(_ds_ref) / _ds_ref.sum()])
SOLS    = [next(s for u0,u1,s in _ZONES if u0<=f<u1) if f<1 else _ZONES[-1][2] for f in _frac]

def rayons_courbure(pts):
    R = np.full(len(pts), np.inf)
    for i in range(1, len(pts)-1):
        dx1,dy1 = pts[i]-pts[i-1]; dx2,dy2 = pts[i+1]-pts[i]
        cross = abs(dx1*dy2 - dy1*dx2)
        chord = np.linalg.norm(pts[i+1]-pts[i-1])
        n1,n2 = np.linalg.norm([dx1,dy1]), np.linalg.norm([dx2,dy2])
        if cross>1e-12 and n1>1e-12 and n2>1e-12:
            R[i] = n1*n2*chord/(2*cross)
    return R

RAYONS = rayons_courbure(CIRCUIT)

# =========================
# MODÈLES PHYSIQUES
# =========================
def eta(K, lam):
    return lam / (2*h*np.sqrt(K*m))

def vmax_virage(R, sol, pneu, ressort):
    if not np.isfinite(R) or R > 300: return V_MAX
    mu      = pneus[pneu]["mu"][sol]
    K, lam  = ressorts[ressort]["K"], ressorts[ressort]["lam"]
    vg      = np.sqrt(mu*g*R)
    A = m*h/(K*l*R); B = 2*h; C = -g*l*R
    disc    = B**2 - 4*A*C
    vr      = np.sqrt((-B+np.sqrt(disc))/(2*A)) * min(1., eta(K,lam)) if disc>=0 else np.inf
    return min(vg, vr, V_MAX)

# Précalcul de toutes les vmax pour chaque config (évite les appels répétés)
def precalc_vmax(pneu, ressort):
    return np.array([vmax_virage(RAYONS[i], SOLS[i], pneu, ressort) for i in range(N_PTS)])

# =========================
# SIMULATION
# =========================
def simuler(pneu, ressort):
    vmax_arr = precalc_vmax(pneu, ressort)
    accels   = np.array([accel(pneu, SOLS[i]) for i in range(N_PTS)])
    ds_arr   = np.array([np.linalg.norm(CIRCUIT[i]-CIRCUIT[i-1]) for i in range(1, N_PTS)])

    v = np.zeros(N_PTS)
    dt = np.zeros(N_PTS - 1)
    for i in range(1, N_PTS):
        ds    = ds_arr[i-1]
        v_new = min(np.sqrt(v[i-1]**2 + 2*accels[i]*ds), vmax_arr[i])
        v[i]  = v_new
        v_moy = (v[i-1] + v[i]) / 2 or 1e-6
        dt[i-1] = ds / v_moy
    return v, dt.sum()

# =========================
# COMPARAISON DES CONFIGS
# =========================
resultats = {}
meilleure_config, meilleur_temps = None, np.inf
for pneu, ressort in itertools_product(pneus, ressorts):
    v_sim, t = simuler(pneu, ressort)
    resultats[(pneu, ressort)] = {
        "v_sim": v_sim, "temps": t,
        "eta":   eta(ressorts[ressort]["K"], ressorts[ressort]["lam"])
    }
    if t < meilleur_temps:
        meilleur_temps, meilleure_config = t, (pneu, ressort)

longueurs = {}
for i in range(N_PTS-1):
    s = SOLS[i]
    longueurs[s] = longueurs.get(s, 0) + np.linalg.norm(CIRCUIT[i+1]-CIRCUIT[i])
longueur_totale = sum(longueurs.values())

# =========================
# AFFICHAGE
# =========================
fig = plt.figure(figsize=(15, 9))
fig.patch.set_facecolor("#1a1a2e")

# Tous les axes commencent au même bas (0.10) et même haut effectif
ax_c = fig.add_axes([0.03, 0.10, 0.55, 0.78])  # circuit (descendu)
ax_p = fig.add_axes([0.62, 0.52, 0.18, 0.43])  # camembert
ax_t = fig.add_axes([0.82, 0.52, 0.16, 0.43])  # tableau
ax_a = fig.add_axes([0.63, 0.22, 0.35, 0.25])  # animation

# Les trois titres alignés à la même hauteur absolue (y=0.935)
TITRE_Y = 0.935
fig.text(0.305, TITRE_Y, 'Circuit — Modélisation du buggy', color='white', fontsize=10, fontweight='bold', ha='center', va='top')
fig.text(0.705, TITRE_Y, 'Répartition du circuit',          color='white', fontsize=10, fontweight='bold', ha='center', va='top')
fig.text(0.900, TITRE_Y, 'Configurations',                  color='white', fontsize=10, fontweight='bold', ha='center', va='top')

for ax in [ax_c, ax_p, ax_t, ax_a]:
    ax.set_facecolor("#16213e")
    for sp in ax.spines.values(): sp.set_edgecolor("#0f3460")

# ---- Circuit (tracé en un seul appel par sol) ----
# Titre géré par fig.text pour alignement avec les autres titres

# Regrouper les segments par sol pour minimiser les appels plot
from itertools import groupby
segs_x = {s: [] for s in couleurs_sol}
segs_y = {s: [] for s in couleurs_sol}
for i in range(N_PTS-1):
    s = SOLS[i]
    segs_x[s] += [CIRCUIT[i,0], CIRCUIT[i+1,0], None]
    segs_y[s] += [CIRCUIT[i,1], CIRCUIT[i+1,1], None]
for s in couleurs_sol:
    if segs_x[s]:
        ax_c.plot(segs_x[s], segs_y[s], color=couleurs_sol[s], linewidth=3, solid_capstyle='round')

# Étiquettes vmax par virage
pneu_b, res_b = meilleure_config
vmax_pts = precalc_vmax(pneu_b, res_b) * 3.6
vmax_pts[vmax_pts >= V_MAX*3.6 - 0.1] = np.nan

in_v, zone, poses = False, [], []
# On ajoute le point de départ dans les positions à éviter dès le début
depart_x, depart_y = CIRCUIT[0]
poses = [(depart_x, depart_y)]   # zone d'exclusion autour du départ

for i in range(N_PTS):
    if np.isfinite(vmax_pts[i]):
        zone.append(i); in_v = True
    else:
        if in_v and len(zone) >= 4:
            mid = zone[len(zone)//2]
            px, py = CIRCUIT[mid]

            # Calcul du décalage pour rester dans les limites du circuit
            x_min, x_max = CIRCUIT[:,0].min(), CIRCUIT[:,0].max()
            y_min, y_max = CIRCUIT[:,1].min(), CIRCUIT[:,1].max()
            marge = 18   # marge en mètres par rapport au bord
            ox, oy = 7, 7
            # Ajuste le décalage si l'étiquette sortirait du cadre
            if px + ox + marge > x_max: ox = -28
            if py + oy + marge > y_max: oy = -14
            if py + oy < y_min + marge: oy = 10

            trop_proche = any(np.sqrt((px-ex)**2+(py-ey)**2) < 35 for ex,ey in poses)
            if not trop_proche:
                ax_c.annotate(f"{vmax_pts[mid]:.1f} km/h", xy=(px,py),
                              xytext=(px+ox, py+oy), fontsize=8, color="white",
                              bbox=dict(boxstyle="round,pad=0.2", facecolor="#e94560", alpha=0.85, edgecolor="none"),
                              arrowprops=dict(arrowstyle="->", color="white", lw=0.8))
                poses.append((px, py))
        zone, in_v = [], False

ax_c.plot(*CIRCUIT[0], "go", markersize=10, zorder=5)
ax_c.annotate("Départ", xy=CIRCUIT[0], xytext=(CIRCUIT[0,0]+5, CIRCUIT[0,1]+7),
              color="lightgreen", fontsize=8,
              arrowprops=dict(arrowstyle="->", color="lightgreen", lw=0.8))
ax_c.axis("equal"); ax_c.grid(color="#0f3460", linestyle="--", alpha=0.5)
ax_c.tick_params(colors="#aaaaaa")
ax_c.set_xlabel("x (m)", color="#aaaaaa"); ax_c.set_ylabel("y (m)", color="#aaaaaa")
patches = [mpatches.Patch(color=c, label=s) for s,c in couleurs_sol.items()]
patches.append(mpatches.Patch(color="#e94560", label="vmax virage"))
ax_c.legend(handles=patches, facecolor="#0f3460", labelcolor="white", fontsize=8)

# ---- Camembert ----
lbl = list(longueurs.keys())
_, _, ats = ax_p.pie([longueurs[s] for s in lbl], labels=lbl,
                      colors=[couleurs_sol[s] for s in lbl],
                      autopct="%1.1f%%", startangle=90,
                      textprops={"color": "white", "fontsize": 8})
for at in ats: at.set_fontsize(7)
ax_p.text(0, -1.55, f"Total : {longueur_totale:.0f} m", ha="center", color="#aaaaaa", fontsize=8)

# ---- Tableau ----
ax_t.axis("off")
rows = []
for (pneu, res), r in resultats.items():
    star = "★" if (pneu, res) == meilleure_config else ""
    rows.append([pneu, res, f"{r['eta']:.2f}", f"{star} {r['temps']:.1f}"])
tbl = ax_t.table(cellText=rows, colLabels=["Pneu", "Ressort", "η", "Temps (s)"],
                  loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(7.5)
for (row, col), cell in tbl.get_celld().items():
    cell.set_facecolor("#0f3460" if row == 0 else "#16213e")
    cell.set_text_props(color="white"); cell.set_edgecolor("#1a1a2e")
    if row > 0 and "★" in str(rows[row-1][3] if col == 3 else ""):
        cell.set_facecolor("#1a4a1a")
tbl.scale(1, 1.3)

# ---- Animation (blit=True + interval plus grand) ----
ax_a.set_title("Vitesse instantanée du Buggy", color="white", fontsize=9, fontweight="bold")
v_kmh = resultats[meilleure_config]["v_sim"] * 3.6
 
# Distance cumulée en mètres pour l'axe x
_ds_cumul = np.concatenate([[0], np.cumsum(
    np.linalg.norm(np.diff(CIRCUIT, axis=0), axis=1)
)])  # shape (N_PTS,)
 
ax_a.set_xlim(0, _ds_cumul[-1]); ax_a.set_ylim(0, v_kmh.max() * 1.15)
ax_a.set_xlabel("Distance (m)", color="#aaaaaa", fontsize=8)
ax_a.set_ylabel("km/h",         color="#aaaaaa", fontsize=8)
ax_a.tick_params(colors="#aaaaaa", labelsize=7)
ax_a.grid(color="#0f3460", linestyle="--", alpha=0.4)
 
line_a, = ax_a.plot([], [], color="#e94560", lw=1.5, animated=True)
dot_a,  = ax_a.plot([], [], "o", color="white", ms=5, animated=True)
txt_a   = ax_a.text(0.86, 0.91, "", transform=ax_a.transAxes, color="white", fontsize=8, animated=True)
car,    = ax_c.plot([], [], "o", color="#e94560", ms=9, zorder=10, animated=True)
 
# Sous-échantillonnage : on saute des frames pour fluidifier
STEP = 3   # valeur initiale, recalculée dynamiquement après
 
def update(f):
    i = f * STEP
    if i >= N_PTS: i = N_PTS - 1
    line_a.set_data(_ds_cumul[:i+1], v_kmh[:i+1])
    dot_a.set_data([_ds_cumul[i]], [v_kmh[i]])
    txt_a.set_text(f"{v_kmh[i]:.1f} km/h")
    car.set_data([CIRCUIT[i,0]], [CIRCUIT[i,1]])
    return line_a, dot_a, txt_a, car

# STEP : on saute des frames pour cibler ~15 fps
# interval calculé pour que 1 tour = meilleur temps réel
STEP = max(1, int(N_PTS / (meilleur_temps * 15)))  # ~15 fps
n_frames = N_PTS // STEP + 1
interval_ms = (meilleur_temps * 1000) / n_frames
ani = FuncAnimation(fig, update, frames=n_frames, interval=interval_ms, blit=True)

# ---- Bandeaux ----
pneu_b, res_b = meilleure_config
eta_b = resultats[meilleure_config]["eta"]

# Bandeau vert aligné sur la largeur du panneau droit uniquement
fig.text(0.820, 0.13,
         f"✦ Meilleure configuration :  Pneu {pneu_b} | Ressort {res_b}",
         color="#ffffff", fontsize=10, fontweight="bold", ha='center',
         bbox=dict(facecolor="#0f6016", edgecolor="#60e945", boxstyle="round,pad=0.4"),
         clip_on=True)

ax_c.text(0.02, 0.98,
          f"Distance : {longueur_totale:.0f} m\nMeilleur Temps : {meilleur_temps:.1f} s",
          transform=ax_c.transAxes, color="white", fontsize=10, fontweight="bold",
          va="top", ha="left",
          bbox=dict(facecolor="#0f3460", edgecolor="#e94560", boxstyle="round,pad=0.4", alpha=0.9))

fig.suptitle("TIPE — Optimisation du buggy télécommandé", color="white",
             fontsize=14, fontweight="bold", y=0.99)
plt.show()