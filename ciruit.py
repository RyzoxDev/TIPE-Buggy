import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================
# PARAMÈTRES PHYSIQUES
# =========================

m = 2.49
g = 9.81

a = 6.0  # accélération (m/s²)
F_frein = 18.0  # force de freinage maximale (N)

h = 0.19
l = 0.30
L_buggy = 0.325

mu = {
    "sable": 0.32,
    "terre": 0.84,
    "beton": 0.67
}

couleurs = {
    "sable": "#d9c27f",
    "terre": "#8b5a2b",
    "beton": "#808080"
}

# =========================
# CIRCUIT
# =========================

circuit = [
    ("ligne", 10, "terre"),
    ("parabole", 8, 8, "sable", "gauche"),
    ("ligne", 10, "terre"),
    ("arc", 5, 2, "terre", "gauche"),
    ("ligne", 25, "beton"),
    ("parabole", 5, 4, "terre", "gauche"),
    ("parabole", 2, 3, "terre", "gauche")
]

# =========================
# MODÈLES
# =========================

def v_glissement(R, mu_val):
    if not np.isfinite(R):
        return np.inf
    return np.sqrt(mu_val * g * R)


def v_retournement(R):
    if not np.isfinite(R):
        return np.inf
    return np.sqrt((g * l * R) / (2 * h))


def vmax_from_R(R, sol):
    vg = v_glissement(R, mu[sol])
    vr = v_retournement(R)
    return min(vg, vr) * 3.6  # km/h


# =========================
# OUTILS GÉOMÉTRIQUES
# =========================

def rotation_locale_vers_monde(x_local, y_local, angle, x0, y0):
    x_world = x0 + x_local * np.cos(angle) - y_local * np.sin(angle)
    y_world = y0 + x_local * np.sin(angle) + y_local * np.cos(angle)
    return x_world, y_world


# =========================
# GÉNÉRATION DU CIRCUIT
# =========================

def generer_points(circuit, n=60):
    x, y, angle = 0.0, 0.0, 0.0
    points = [(x, y)]
    sols = []
    vmax_local = [np.nan]

    for segment in circuit:
        typ = segment[0]

        if typ == "ligne":
            _, longueur, sol = segment

            for _ in range(n):
                dx = (longueur / n) * np.cos(angle)
                dy = (longueur / n) * np.sin(angle)
                x += dx
                y += dy
                points.append((x, y))
                sols.append(sol)
                vmax_local.append(np.nan)

        elif typ == "arc":
            _, longueur, R, sol, sens = segment

            signe = 1 if sens == "gauche" else -1
            theta_total = signe * longueur / R
            theta_vals = np.linspace(0, theta_total, n + 1)[1:]

            xc = x - signe * R * np.sin(angle)
            yc = y + signe * R * np.cos(angle)

            v_lim = vmax_from_R(R, sol)

            for dtheta in theta_vals:
                xi = xc + signe * R * np.sin(angle + dtheta)
                yi = yc - signe * R * np.cos(angle + dtheta)
                points.append((xi, yi))
                sols.append(sol)
                vmax_local.append(v_lim)

            x, y = points[-1]
            angle += theta_total

        elif typ == "demi_cercle":
            _, R, sol, sens = segment

            signe = 1 if sens == "gauche" else -1
            theta_total = signe * np.pi
            theta_vals = np.linspace(0, theta_total, n + 1)[1:]

            xc = x - signe * R * np.sin(angle)
            yc = y + signe * R * np.cos(angle)

            v_lim = vmax_from_R(R, sol)

            for dtheta in theta_vals:
                xi = xc + signe * R * np.sin(angle + dtheta)
                yi = yc - signe * R * np.cos(angle + dtheta)
                points.append((xi, yi))
                sols.append(sol)
                vmax_local.append(v_lim)

            x, y = points[-1]
            angle += theta_total

        elif typ == "parabole":
            _, longueur, fleche, sol, sens = segment

            signe = 1 if sens == "gauche" else -1

            x_loc = np.linspace(0, longueur, n + 1)[1:]
            y_loc = signe * fleche * (x_loc / longueur) ** 2

            for xl, yl in zip(x_loc, y_loc):
                xi, yi = rotation_locale_vers_monde(xl, yl, angle, x, y)
                points.append((xi, yi))
                sols.append(sol)

                yp = signe * 2 * fleche * xl / (longueur ** 2)
                ypp = signe * 2 * fleche / (longueur ** 2)

                if abs(ypp) < 1e-12:
                    R_local = np.inf
                else:
                    R_local = (1 + yp**2) ** 1.5 / abs(ypp)

                vmax_local.append(vmax_from_R(R_local, sol))

            x, y = points[-1]
            pente_fin = signe * 2 * fleche / longueur
            angle += np.arctan(pente_fin)

        else:
            raise ValueError(f"Type de segment inconnu : {typ}")

    points.append(points[0])
    sols.append(sols[0] if len(sols) > 0 else "beton")
    vmax_local.append(np.nan)

    return np.array(points), sols, np.array(vmax_local)


points, sols, vmax_local = generer_points(circuit, n=60)

# =========================
# ACCÉLÉRATION / FREINAGE
# =========================

a_frein = F_frein / m  # m/s²

# Passe 1 : accélération seule
v_forward = [0.0]  # m/s

for i in range(1, len(points)):
    dx = points[i][0] - points[i - 1][0]
    dy = points[i][1] - points[i - 1][1]
    ds = np.sqrt(dx**2 + dy**2)

    v_prev = v_forward[-1]
    v_new = np.sqrt(v_prev**2 + 2 * a * ds)

    if np.isfinite(vmax_local[i]):
        v_new = min(v_new, vmax_local[i] / 3.6)

    v_forward.append(v_new)

v_forward = np.array(v_forward)

# Passe 2 : freinage anticipé
v_reel = np.copy(v_forward)

for i in range(len(points) - 2, -1, -1):
    dx = points[i + 1][0] - points[i][0]
    dy = points[i + 1][1] - points[i][1]
    ds = np.sqrt(dx**2 + dy**2)

    # vitesse max au point i pour pouvoir freiner jusqu'au point i+1
    v_lim_frein = np.sqrt(max(v_reel[i + 1]**2 + 2 * a_frein * ds, 0.0))
    v_reel[i] = min(v_reel[i], v_lim_frein)

    if np.isfinite(vmax_local[i]):
        v_reel[i] = min(v_reel[i], vmax_local[i] / 3.6)

# =========================
# TEMPS ET DISTANCE DE FREINAGE
# =========================

temps_total = 0.0
temps_freinage_total = 0.0
distance_freinage_totale = 0.0

en_freinage = np.zeros(len(points), dtype=bool)

for i in range(1, len(points)):
    dx = points[i][0] - points[i - 1][0]
    dy = points[i][1] - points[i - 1][1]
    ds = np.sqrt(dx**2 + dy**2)

    v1 = v_reel[i - 1]
    v2 = v_reel[i]

    v_moy = max((v1 + v2) / 2, 1e-6)
    temps_total += ds / v_moy

    if v2 < v1:
        dt_frein = (v1 - v2) / a_frein
        temps_freinage_total += dt_frein
        distance_freinage_totale += ds
        en_freinage[i] = True

v_reel_kmh = v_reel * 3.6

print(f"Décélération de freinage : {a_frein:.2f} m/s²")
print(f"Temps total du tour : {temps_total:.2f} s")
print(f"Temps total de freinage : {temps_freinage_total:.2f} s")
print(f"Distance totale de freinage : {distance_freinage_totale:.2f} m")

# =========================
# AFFICHAGE CIRCUIT
# =========================

plt.figure(figsize=(10, 7))

for i in range(len(points) - 1):
    x_seg = [points[i][0], points[i + 1][0]]
    y_seg = [points[i][1], points[i + 1][1]]
    plt.plot(x_seg, y_seg, color=couleurs[sols[i]], linewidth=3)

# afficher quelques vitesses max locales
for i in range(0, len(points), 35):
    if np.isfinite(vmax_local[i]):
        plt.text(points[i][0], points[i][1], f"{vmax_local[i]:.1f} km/h", fontsize=8)

# surligner les zones de freinage
for i in range(1, len(points)):
    if en_freinage[i]:
        x_seg = [points[i - 1][0], points[i][0]]
        y_seg = [points[i - 1][1], points[i][1]]
        plt.plot(x_seg, y_seg, color="red", linewidth=4, alpha=0.6)

plt.axis('equal')
plt.title("Circuit avec accélération et freinage")
plt.grid()

# =========================
# ANIMATION
# =========================

car, = plt.plot([], [], 'bo')
text_v = plt.text(0, 0, '')
text_info = plt.text(
    0.02, 0.95,
    "",
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.7)
)

def update(frame):
    x, y = points[frame]
    car.set_data([x], [y])
    text_v.set_position((x, y))
    text_v.set_text(f"{v_reel_kmh[frame]:.1f} km/h")

    if en_freinage[frame]:
        text_info.set_text("Phase : freinage")
    else:
        text_info.set_text("Phase : accélération / maintien")

    return car, text_v, text_info

ani = FuncAnimation(plt.gcf(), update, frames=len(points), interval=30, blit=False)

plt.show()