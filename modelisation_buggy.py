# -*- coding: utf-8 -*-
"""
TIPE — Optimisation du buggy telecommande
Modelisation physique : pneus x ressorts x sols
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from itertools import product as itertools_product

# ============================================================
# 1. PARAMETRES PHYSIQUES
# ============================================================
m     = 2.492        # masse (kg)
g     = 9.81         # gravite (m/s²)
h     = 0.19         # hauteur centre de gravite (m)
l     = 0.30         # demi-voie (m)
V_MAX = 60 / 3.6     # vitesse max moteur (m/s)
A_MOTEUR = 7.4

# ============================================================
# 2. DONNEES EXPERIMENTALES
# ============================================================
# Sensibilite de l'amortissement selon le sol
# alpha proche de 0 : lambda peu important | alpha proche de 1 : lambda tres important
ALPHA_SOL = {"beton": 0.1, "terre": 0.4, "sable": 0.9}

PNEUS = {
    "slick":  {"mu": {"sable": 0.28, "terre": 0.75, "beton": 0.85}},
    "sillon": {"mu": {"sable": 0.42, "terre": 0.84, "beton": 0.67}},
}
RESSORTS = {
    "souple": {"K": 388, "lam": 62.2},
    "rigide": {"K": 927, "lam": 96.1},
}
LAM_REF = max(r["lam"] for r in RESSORTS.values())  # normalisation = ressort le plus rigide

COULEURS_SOL = {"sable": "#d9c27f", "terre": "#8b5a2b", "beton": "#808080"}

# ============================================================
# 3. MODELES PHYSIQUES
# ============================================================
def mu_eff(mu, sol, lam):
    """
    Adherence effective tenant compte de l'amortissement et du sol.
    
    Formule : mu_eff = mu * (1 - alpha * exp(-lam / lam_ref))
    
    Physique :
      - lam grand (ressort rigide) → exp ≈ 0 → mu_eff ≈ mu   (pas de penalite)
      - lam faible (ressort souple) → exp ≈ 1 → penalite alpha*mu
      - Sur beton (alpha=0.1) : lambda peu important
      - Sur sable (alpha=0.9) : lambda tres important
    """
    return mu * (1.0 - ALPHA_SOL[sol] * np.exp(-lam / LAM_REF))

def a_reelle(pneu, sol, ressort):
    """
    Acceleration reelle = min(a_moteur, mu_eff * g)
    - Limite moteur : a_moteur
    - Limite adherence : mu_eff * g (roues qui patinent)
    """
    lam = RESSORTS[ressort]["lam"]
    mu  = mu_eff(PNEUS[pneu]["mu"][sol], sol, lam)
    return min(A_MOTEUR, mu * g)

def vmax_virage(R, sol, pneu, ressort):
    """Vitesse max en virage (adherence + renversement)."""
    if not np.isfinite(R) or R > 300:
        return V_MAX
    lam = RESSORTS[ressort]["lam"]
    K   = RESSORTS[ressort]["K"]
    mu  = mu_eff(PNEUS[pneu]["mu"][sol], sol, lam)
    vg  = np.sqrt(mu * g * R)                          # limite adherence laterale
    A   = m * h / (K * l * R)
    disc = (2*h)**2 + 4 * A * g * l * R
    vr  = np.sqrt((-2*h + np.sqrt(disc)) / (2*A))      # limite renversement
    return min(vg, vr, V_MAX)

# ============================================================
# 4. CIRCUIT — Catmull-Rom paramétrique ferme
# ============================================================
def catmull_rom_ferme(kp, n=50):
    """Spline Catmull-Rom fermee passant par tous les points de controle."""
    def tj(ti, a, b):
        return ti + ((b[0]-a[0])**2 + (b[1]-a[1])**2) ** 0.5

    def segment(p0, p1, p2, p3):
        t0 = 0.
        t1 = tj(t0, p0, p1); t2 = tj(t1, p1, p2); t3 = tj(t2, p2, p3)
        pts = []
        for t in np.linspace(t1, t2, n, endpoint=False):
            A1 = (t1-t)/(t1-t0)*np.array(p0) + (t-t0)/(t1-t0)*np.array(p1)
            A2 = (t2-t)/(t2-t1)*np.array(p1) + (t-t1)/(t2-t1)*np.array(p2)
            A3 = (t3-t)/(t3-t2)*np.array(p2) + (t-t2)/(t3-t2)*np.array(p3)
            B1 = (t2-t)/(t2-t0)*A1 + (t-t0)/(t2-t0)*A2
            B2 = (t3-t)/(t3-t1)*A2 + (t-t1)/(t3-t1)*A3
            pts.append((t2-t)/(t2-t1)*B1 + (t-t1)/(t2-t1)*B2)
        return np.array(pts)

    N   = len(kp)
    pts = np.vstack([segment(kp[(i-1)%N], kp[i], kp[(i+1)%N], kp[(i+2)%N]) for i in range(N)])
    return np.vstack([pts, pts[0]])

# Points de controle normalises [0,1]
_KP_NORM = np.array([
    [0.50, 0.50],  # centre départ
    [0.42, 0.60],  # diagonale montée gauche
    [0.30, 0.72],  # montée épingle haut-gauche
    [0.20, 0.80],  # épingle haut-gauche entrée
    [0.13, 0.74],  # épingle haut-gauche sommet
    [0.16, 0.64],  # épingle haut-gauche sortie
    [0.22, 0.55],  # grande courbe gauche haut
    [0.18, 0.44],  # grande courbe gauche milieu
    [0.22, 0.33],  # grande courbe gauche bas
    [0.16, 0.24],  # épingle bas-gauche entrée
    [0.13, 0.15],  # épingle bas-gauche sommet
    [0.22, 0.10],  # épingle bas-gauche sortie
    [0.35, 0.14],  # ligne droite bas
    [0.50, 0.18],  # bas milieu
    [0.62, 0.22],  # bas droite
    [0.72, 0.28],  # virage bas-droite
    [0.80, 0.38],  # chicane droite entrée
    [0.86, 0.46],  # chicane droite sommet
    [0.80, 0.54],  # chicane droite creux
    [0.86, 0.62],  # chicane droite 2 sommet
    [0.78, 0.70],  # chicane droite 2 sortie
    [0.68, 0.62],  # retour diagonal haut
    [0.60, 0.56],  # ligne diagonale retour
])

# Mise a l'echelle → longueur totale ≈ 500 m
_tmp = catmull_rom_ferme(_KP_NORM.tolist())
_S   = 500 / np.linalg.norm(np.diff(_tmp, axis=0), axis=1).sum()
CIRCUIT = catmull_rom_ferme((_KP_NORM * _S).tolist())
N_PTS   = len(CIRCUIT)

# Zones de sol par fraction de circuit parcouru
_ZONES = [
    (0.00, 0.18, "beton"),
    (0.18, 0.65, "terre"),
    (0.65, 0.85, "beton"),
    (0.85, 1.00, "beton"),
]
_ds   = np.linalg.norm(np.diff(CIRCUIT, axis=0), axis=1)
_frac = np.concatenate([[0], np.cumsum(_ds) / _ds.sum()])
SOLS  = [next(s for u0, u1, s in _ZONES if u0 <= f < u1) if f < 1 else _ZONES[-1][2]
         for f in _frac]

# Rayons de courbure vectorises
def rayons_courbure(pts):
    R = np.full(len(pts), np.inf)
    d1 = pts[1:-1] - pts[:-2]
    d2 = pts[2:]   - pts[1:-1]
    cross = np.abs(d1[:,0]*d2[:,1] - d1[:,1]*d2[:,0])
    chord = np.linalg.norm(pts[2:] - pts[:-2], axis=1)
    n1    = np.linalg.norm(d1, axis=1)
    n2    = np.linalg.norm(d2, axis=1)
    mask  = (cross > 1e-12) & (n1 > 1e-12) & (n2 > 1e-12)
    R[1:-1][mask] = n1[mask] * n2[mask] * chord[mask] / (2 * cross[mask])
    return R

RAYONS = rayons_courbure(CIRCUIT)
DS_ARR = np.linalg.norm(np.diff(CIRCUIT, axis=0), axis=1)  # distances inter-points
DS_CUM = np.concatenate([[0], np.cumsum(DS_ARR)])           # distance cumulee

# ============================================================
# 5. SIMULATION
# ============================================================
def simuler(pneu, ressort):
    vmax_arr = np.array([vmax_virage(RAYONS[i], SOLS[i], pneu, ressort) for i in range(N_PTS)])
    acc_arr  = np.array([a_reelle(pneu, SOLS[i], ressort) for i in range(N_PTS)])

    v  = np.zeros(N_PTS)
    dt = np.zeros(N_PTS - 1)
    for i in range(1, N_PTS):
        v[i]    = min(np.sqrt(v[i-1]**2 + 2 * acc_arr[i] * DS_ARR[i-1]), vmax_arr[i])
        v_moy   = (v[i-1] + v[i]) / 2 or 1e-9
        dt[i-1] = DS_ARR[i-1] / v_moy
    return v, dt.sum()

# ============================================================
# 6. COMPARAISON DES CONFIGURATIONS
# ============================================================
resultats = {}
meilleure_config, meilleur_temps = None, np.inf

for pneu, ressort in itertools_product(PNEUS, RESSORTS):
    v_sim, t = simuler(pneu, ressort)
    resultats[(pneu, ressort)] = {"v_sim": v_sim, "temps": t}
    if t < meilleur_temps:
        meilleur_temps, meilleure_config = t, (pneu, ressort)

longueurs = {}
for i in range(N_PTS - 1):
    s = SOLS[i]
    longueurs[s] = longueurs.get(s, 0) + DS_ARR[i]
longueur_totale = sum(longueurs.values())

# ============================================================
# 7. AFFICHAGE GRAPHIQUE
# ============================================================
fig = plt.figure(figsize=(15, 9))
fig.patch.set_facecolor("#1a1a2e")

ax_c = fig.add_axes([0.03, 0.10, 0.55, 0.78])   # circuit
ax_p = fig.add_axes([0.62, 0.54, 0.18, 0.43])   # camembert
ax_t = fig.add_axes([0.82, 0.54, 0.16, 0.43])   # tableau
ax_a = fig.add_axes([0.63, 0.20, 0.35, 0.36])   # vitesse animee

for ax in [ax_c, ax_p, ax_t, ax_a]:
    ax.set_facecolor("#16213e")
    for sp in ax.spines.values():
        sp.set_edgecolor("#0f3460")

TITRE_Y = 0.93
fig.text(0.305, TITRE_Y, "Circuit — Modelisation du buggy",
         color="white", fontsize=10, fontweight="bold", ha="center", va="top")
fig.text(0.710, TITRE_Y, "Repartition du circuit",
         color="white", fontsize=10, fontweight="bold", ha="center", va="top")
fig.text(0.900, TITRE_Y, "Configurations",
         color="white", fontsize=10, fontweight="bold", ha="center", va="top")
fig.suptitle("TIPE — Optimisation du buggy telecommande",
             color="white", fontsize=13, fontweight="bold", y=0.98)

# -- Circuit colore par sol --
segs = {s: ([], []) for s in COULEURS_SOL}
for i in range(N_PTS - 1):
    s = SOLS[i]
    segs[s][0].extend([CIRCUIT[i,0], CIRCUIT[i+1,0], None])
    segs[s][1].extend([CIRCUIT[i,1], CIRCUIT[i+1,1], None])
for s, (sx, sy) in segs.items():
    if sx:
        ax_c.plot(sx, sy, color=COULEURS_SOL[s], linewidth=3, solid_capstyle="round")

# -- Annotations vmax virages --
pneu_b, res_b = meilleure_config
vmax_pts = precalc = np.array([
    vmax_virage(RAYONS[i], SOLS[i], pneu_b, res_b) * 3.6 for i in range(N_PTS)
])
vmax_pts[vmax_pts >= V_MAX * 3.6 - 0.1] = np.nan

x_min, x_max = CIRCUIT[:,0].min(), CIRCUIT[:,0].max()
y_min, y_max = CIRCUIT[:,1].min(), CIRCUIT[:,1].max()
in_v, zone, poses = False, [], [tuple(CIRCUIT[0])]

for i in range(N_PTS):
    if np.isfinite(vmax_pts[i]):
        zone.append(i); in_v = True
    else:
        if in_v and len(zone) >= 4:
            mid = zone[len(zone)//2]
            px, py = CIRCUIT[mid]
            ox, oy = 7, 7
            if px + ox + 18 > x_max: ox = -28
            if py + oy + 18 > y_max: oy = -14
            if py + oy < y_min + 18:  oy = 10
            if not any(np.hypot(px-ex, py-ey) < 35 for ex, ey in poses):
                ax_c.annotate(f"{vmax_pts[mid]:.1f} km/h", xy=(px, py),
                              xytext=(px+ox, py+oy), fontsize=8, color="white",
                              bbox=dict(boxstyle="round,pad=0.2", facecolor="#e94560",
                                        alpha=0.85, edgecolor="none"),
                              arrowprops=dict(arrowstyle="->", color="white", lw=0.8))
                poses.append((px, py))
        zone, in_v = [], False

ax_c.plot(*CIRCUIT[0], "go", markersize=10, zorder=5)
ax_c.annotate("Depart", xy=CIRCUIT[0], xytext=(CIRCUIT[0,0]+5, CIRCUIT[0,1]+7),
              color="lightgreen", fontsize=8,
              arrowprops=dict(arrowstyle="->", color="lightgreen", lw=0.8))
ax_c.axis("equal")
ax_c.grid(color="#0f3460", linestyle="--", alpha=0.5)
ax_c.tick_params(colors="#aaaaaa")
ax_c.set_xlabel("x (m)", color="#aaaaaa")
ax_c.set_ylabel("y (m)", color="#aaaaaa")
patches = [mpatches.Patch(color=c, label=s) for s, c in COULEURS_SOL.items()]
patches.append(mpatches.Patch(color="#e94560", label="vmax virage"))
ax_c.legend(handles=patches, facecolor="#0f3460", labelcolor="white", fontsize=8)
ax_c.text(0.02, 0.98, f"Distance : {longueur_totale:.0f} m\nMeilleur Temps : {meilleur_temps:.2f} s",
          transform=ax_c.transAxes, color="white", fontsize=10, fontweight="bold",
          va="top", ha="left",
          bbox=dict(facecolor="#0f3460", edgecolor="#e94560", boxstyle="round,pad=0.4", alpha=0.9))

# -- Camembert --
lbl = list(longueurs.keys())
_, _, ats = ax_p.pie([longueurs[s] for s in lbl], labels=lbl,
                     colors=[COULEURS_SOL[s] for s in lbl],
                     autopct="%1.1f%%", startangle=90,
                     textprops={"color": "white", "fontsize": 8})
for at in ats: at.set_fontsize(7)
ax_p.text(0, -1.55, f"Total : {longueur_totale:.0f} m",
          ha="center", color="#aaaaaa", fontsize=8)

# -- Tableau --
ax_t.axis("off")
rows = []
for (pneu, res), r in resultats.items():
    star = "* " if (pneu, res) == meilleure_config else "  "
    rows.append([pneu, res, f"{star}{r['temps']:.2f}"])
tbl = ax_t.table(cellText=rows, colLabels=["Pneu", "Ressort", "Temps (s)"],
                 loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(7.5)
for (row, col), cell in tbl.get_celld().items():
    is_best = row > 0 and rows[row-1][2].startswith("* ")
    cell.set_facecolor("#0f3460" if row == 0 else ("#1a4a1a" if is_best else "#16213e"))
    cell.set_text_props(color="white")
    cell.set_edgecolor("#1a1a2e")
tbl.scale(1, 1.3)

# -- Animation vitesse --
ax_a.set_title("Vitesse instantanee — Meilleure configuration", color="white", fontsize=9, fontweight="bold")
v_kmh = resultats[meilleure_config]["v_sim"] * 3.6
ax_a.set_xlim(0, DS_CUM[-1]); ax_a.set_ylim(0, v_kmh.max() * 1.15)
ax_a.set_xlabel("Distance (m)", color="#aaaaaa", fontsize=8)
ax_a.set_ylabel("km/h",         color="#aaaaaa", fontsize=8)
ax_a.tick_params(colors="#aaaaaa", labelsize=7)
ax_a.grid(color="#0f3460", linestyle="--", alpha=0.4)

line_a, = ax_a.plot([], [], color="#e94560", lw=1.5, animated=True)
dot_a,  = ax_a.plot([], [], "o", color="white", ms=5, animated=True)
txt_a   = ax_a.text(0.97, 0.93, "", transform=ax_a.transAxes, ha="right",
                    color="white", fontsize=9, fontweight="bold", animated=True)
car,    = ax_c.plot([], [], "o", color="#e94560", ms=9, zorder=10, animated=True)

STEP       = max(1, int(N_PTS / (meilleur_temps * 15)))
n_frames   = N_PTS // STEP + 1
interval_ms = meilleur_temps * 1000 / n_frames

def update(f):
    i = min(f * STEP, N_PTS - 1)
    line_a.set_data(DS_CUM[:i+1], v_kmh[:i+1])
    dot_a.set_data([DS_CUM[i]], [v_kmh[i]])
    txt_a.set_text(f"{v_kmh[i]:.1f} km/h")
    car.set_data([CIRCUIT[i,0]], [CIRCUIT[i,1]])
    return line_a, dot_a, txt_a, car

ani = FuncAnimation(fig, update, frames=n_frames, interval=interval_ms, blit=True)

fig.text(0.820, 0.11,
         f"Meilleure config : Pneu {pneu_b} | Ressort {res_b} | {meilleur_temps:.2f} s",
         color="white", fontsize=9, fontweight="bold", ha="center",
         bbox=dict(facecolor="#0f6016", edgecolor="#60e945", boxstyle="round,pad=0.4"))

plt.show()