"""
Fig. 7 -- contact energy and normal force: original IMC vs. squared softplus (ours).

Top row   : IMC [Choi et al., J. Appl. Mech. 2021, Eq. (7)], written in terms of
            the signed clearance d (d = Delta - 2h) with smoothing length
            delta = 1/c_ek (clearance units):
                phi_IMC(d) = k_c * delta * log(1 + exp(-d/delta))
                f_n        = -dphi/dd = k_c * sigmoid(-d/delta)
            delta -> 0 : linear penalty k_c*max(-d, 0), force = k_c * step.
Bottom row: squared softplus, Eq. (41)/(43) of the manuscript
                phi(d) = k_c/2 * [delta * log(1 + exp(-d/delta))]^2
                f_n    = k_c * h * sigma,   h = delta*log(1+e^{-d/delta})
            delta -> 0 : quadratic penalty k_c/2 * max(-d, 0)^2.

Output: fig7_contact_energy.pdf / .svg (single column, editable text).
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "source"))
import tro_style as ts  # noqa: E402

ts.apply()

# ------------------------------------------------------------------ parameters
DELTAS = [0.35, 0.80]                 # delta / ell_star
COLORS = [ts.C["blue"], ts.C["orange"]]
D = np.linspace(-3.0, 3.0, 1201)      # d / ell_star (k_c = ell_star = 1)


def softplus(d, delta):
    """delta * log(1 + exp(-d/delta)), numerically stable."""
    return delta * np.logaddexp(0.0, -d / delta)


def sigmoid_neg(d, delta):
    """sigma = 1 / (1 + exp(d/delta))."""
    return 0.5 * (1.0 - np.tanh(d / (2.0 * delta)))


# IMC (linear softplus)
def imc_energy(d, delta):  return softplus(d, delta)
def imc_force(d, delta):   return sigmoid_neg(d, delta)
def imc_energy0(d):        return np.maximum(-d, 0.0)
def imc_force0(d):         return (d < 0).astype(float)

# Ours (squared softplus)
def sq_energy(d, delta):   return 0.5 * softplus(d, delta) ** 2
def sq_force(d, delta):    return softplus(d, delta) * sigmoid_neg(d, delta)
def sq_energy0(d):         return 0.5 * np.maximum(-d, 0.0) ** 2
def sq_force0(d):          return np.maximum(-d, 0.0)


ROWS = [
    dict(name="IMC",
         E=imc_energy, F=imc_force, E0=imc_energy0, F0=imc_force0,
         ylab_E=r"$\phi^{\mathrm{IMC}}_\delta/(k_c\ell_{⋆})$",
         ylab_F=r"$f_n/k_c$",
         ylim_E=(0, 3.3), ylim_F=(0, 1.15),
         yt_E=[0, 1, 2, 3], yt_F=[0, 0.5, 1]),
    dict(name="Ours",
         E=sq_energy, F=sq_force, E0=sq_energy0, F0=sq_force0,
         ylab_E=r"$\phi_\delta/(k_c\ell_{⋆}^2)$",
         ylab_F=r"$f_n/(k_c\ell_{⋆})$",
         ylim_E=(0, 5.0), ylim_F=(0, 3.4),
         yt_E=[0, 2, 4], yt_F=[0, 1, 2, 3]),
]

# ------------------------------------------------------------------ figure
fig, axs = plt.subplots(2, 2, figsize=ts.figsize(ts.COL_W, 2.75),
                        sharex=True)
letters = iter("abcd")

for r, row in enumerate(ROWS):
    for c, kind in enumerate(("energy", "normal force")):
        ax = axs[r, c]
        f = row["E"] if c == 0 else row["F"]
        f0 = row["E0"] if c == 0 else row["F0"]
        ylim = row["ylim_E"] if c == 0 else row["ylim_F"]

        ax.axvspan(D[0], 0.0, color=ts.C["shade"], lw=0, zorder=0)
        ax.axvline(0.0, color=ts.C["gray"], lw=ts.LW_AXES, ls=":", zorder=1)
        for delta, col in zip(DELTAS, COLORS):
            ax.plot(D, f(D, delta), color=col, lw=ts.LW_DATA, zorder=3)
        if c == 1 and r == 0:          # draw the step with a vertical jump
            ax.plot([D[0], 0, 0, D[-1]], [1, 1, 0, 0], color=ts.C["gray"],
                    lw=ts.LW_REF, ls="--", zorder=2)
        else:
            ax.plot(D, f0(D), color=ts.C["gray"], lw=ts.LW_REF, ls="--", zorder=2)

        ax.set_xlim(D[0], D[-1])
        ax.set_ylim(*ylim)
        ax.set_xticks([-3, 0, 3])
        ax.set_yticks(row["yt_E"] if c == 0 else row["yt_F"])
        ax.set_ylabel(row["ylab_E"] if c == 0 else row["ylab_F"], labelpad=2)
        if r == 1:
            ax.set_xlabel(r"$d/\ell_{⋆}$", labelpad=1)
        ts.panel_title(ax, next(letters), f"{row['name']}: {kind}")

# shared legend above the panels
handles = [Line2D([], [], color=col, lw=ts.LW_DATA) for col in COLORS]
handles.append(Line2D([], [], color=ts.C["gray"], lw=ts.LW_REF, ls="--"))
labels = [r"$\delta/\ell_{⋆} = " + f"{d:.2f}$" for d in DELTAS] + [r"$\delta \to 0$"]
fig.legend(handles, labels, loc="upper center", ncol=3,
           bbox_to_anchor=(0.5, 1.0), borderaxespad=0.1)

fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=0.6, w_pad=1.0)

if __name__ == "__main__":
    for p in ts.save(fig, os.path.join(HERE, "fig7_contact_energy")):
        print("saved", p)
