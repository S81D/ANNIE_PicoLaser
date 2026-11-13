"""
Single-PMT laser timing spectrum with a Gaussian fit to the prompt peak.

Recreates one panel of `exemplary_PMT_timing_fits_laser.png` for a
well-behaved barrel PMT (sharp prompt peak, limited late-light tail),
following the fit recipe in S81D/ANNIE_PicoLaser
(2023_PMT_Timing_Analysis.ipynb, cell 8).

Differences from the collaboration analysis:
  * that analysis plots the hit-time residual  t_res = t - TOF - t0,
    with t0 an arbitrary constant and TOF the light travel time from the
    central laser source to each PMT.  This file only stores per-hit
    times, so TOF (a near-constant per PMT for a central source) is not
    available.  Instead we chain to  t0 = earliest recorded hit  for the
    PMT, i.e. the x-axis is (t - t0); shape and sigma are unchanged, only
    the peak position shifts.
  * single panel, run number stripped from the title.
  * displayed at 0.5 ns bins; the fit-window recipe still runs on the
    1 ns bins its 0.02 threshold was tuned for (FIT_BIN_SIZE).
  * the Gaussian is drawn only across its central ~80% (CURVE_COVER).

Kept metrics: total hits, fit mean (mu), fit width (sigma).

Env: physics  (uproot 5.7, numpy 2.5, scipy 1.18, matplotlib 3.11)
"""

import numpy as np
import uproot
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm

ROOT_FILE  = "laser_data_singlePE.root"
PMT        = "PMT_411"     # barrel PMT, clean prompt peak (auto-fit in the original)
LOCATION   = "Barrel"
BIN_SIZE     = 0.5         # ns, display binning
FIT_BIN_SIZE = 1.0         # ns, coarser binning the window recipe was tuned for
THRESHOLD  = 0.02          # density level that defines the start of the fit window
CURVE_COVER = 0.80         # fraction of the Gaussian area to actually draw (~+/-1.28 sigma)
OUT_STEM   = "PMT_timing_fit_laser_single"

CURVE_COLOR = "dodgerblue"


# --------------------------------------------------------------------------
# load hits, reference to a common t0 (earliest hit for this PMT)
# --------------------------------------------------------------------------
hits = uproot.open(ROOT_FILE)[PMT]["time_hist"].array(library="np")
n_hits = hits.size

t0 = hits.min()
dt = hits - t0

# --------------------------------------------------------------------------
# display histogram, normalised to unit area (density=True).
# Edges are offset by half a bin so that the heavily-populated integer- and
# half-integer-ns timestamps (the DAQ clock quantises hard onto these) sit at
# bin centres rather than straddling an edge -- otherwise sub-ns bins alias
# into a sawtooth.
# --------------------------------------------------------------------------
binning   = np.arange(-BIN_SIZE / 2, np.ceil(dt.max()) + BIN_SIZE, BIN_SIZE)
bin_edges = binning[:-1]                       # left edge of each bin
counts, _ = np.histogram(dt, bins=binning, density=True)
peak_density = counts.max()

# --------------------------------------------------------------------------
# fit window (original recipe, on 1 ns bins the 0.02 threshold was tuned for):
# symmetric about the peak bin, from one bin below the first bin that clears
# THRESHOLD up to the mirror point (+1 extra ns on the high side).
# --------------------------------------------------------------------------
fit_binning = np.arange(0.0, np.ceil(dt.max()) + FIT_BIN_SIZE, FIT_BIN_SIZE)
fit_edges   = fit_binning[:-1]
fit_counts, _ = np.histogram(dt, bins=fit_binning, density=True)

peak_edge = fit_edges[np.argmax(fit_counts)]                  # left edge of tallest bin
over = np.flatnonzero(fit_counts > THRESHOLD)
lo = fit_edges[over[0]] - FIT_BIN_SIZE                        # start of fit window
half = peak_edge - lo
hi = peak_edge + half + FIT_BIN_SIZE                          # end of fit window

# --------------------------------------------------------------------------
# Gaussian fit to the prompt peak: unbinned MLE on the windowed hits
# (scipy norm.fit -> sample mean / std), same as the original.
# --------------------------------------------------------------------------
core = dt[(dt > lo) & (dt < hi)]
mu, sigma = norm.fit(core)
quality = core.size / n_hits          # fraction of hits inside the fit window

print(f"{PMT}:  N = {n_hits:,}   t0 = {t0:.2f} ns   window = [{lo:.0f}, {hi:.0f}] ns")
print(f"   mu    = {mu:.2f} ns   (in t - t0 frame)")
print(f"   sigma = {sigma:.2f} ns")
print(f"   fit quality (hits in window / total) = {quality:.2f}")
print(f"   full-sample mean/std of (t - t0)     = {dt.mean():.2f} / {dt.std():.2f} ns")

# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------
mpl.rc("font", family="serif", size=12)
mpl.rcParams["mathtext.fontset"] = "cm"        # Computer Modern math, as in the original

fig, ax = plt.subplots(figsize=(4.5, 3.9), constrained_layout=True)

# black density histogram
ax.hist(bin_edges, binning, weights=counts, color="black")

# fitted Gaussian, drawn only across the central CURVE_COVER of its area
# (~+/-1.28 sigma).  mu, sigma are fixed from the unbinned fit; the height is
# a 1-parameter least-squares match to the displayed bins over that same
# range, so the curve overlays the peak rather than floating above one
# comb-quantised bin.
centers = bin_edges + BIN_SIZE / 2
xcap = norm.ppf(0.5 + CURVE_COVER / 2) * sigma        # half-width of the drawn arc
in_arc = np.abs(centers - mu) <= xcap
shape = np.exp(-0.5 * ((centers[in_arc] - mu) / sigma) ** 2)
amp = np.sum(counts[in_arc] * shape) / np.sum(shape ** 2)

xx = np.linspace(mu - xcap, mu + xcap, 300)
ax.plot(xx, amp * np.exp(-0.5 * ((xx - mu) / sigma) ** 2), color=CURVE_COLOR, lw=1.4,
        label=fr"$\mu = {mu:.2f}$ ns" + "\n" + fr"$\sigma = {sigma:.2f}$ ns")
ax.axvline(mu, color=CURVE_COLOR, lw=1.0, ls="dashed")

ax.set_xlim(0, 23)
ax.set_ylim(0, 1.18 * peak_density)
ax.set_xlabel(r"$t - t_0$  [ns]")
ax.set_ylabel(r"Fraction of hits / ns")
ax.set_title(f"PMT {PMT.split('_')[1]} ({LOCATION})", pad=8)

ax.minorticks_on()
ax.tick_params(which="both", direction="in", top=True, right=True)
ax.tick_params(which="major", length=4)
ax.tick_params(which="minor", length=2)

ax.legend(frameon=False, loc="upper right", fontsize=11, handlelength=1.3)
ax.text(0.97, 0.60, f"{n_hits:,} hits", transform=ax.transAxes,
        ha="right", va="top", fontsize=9)

fig.savefig(f"{OUT_STEM}.png", dpi=300)
fig.savefig(f"{OUT_STEM}.pdf")
print(f"wrote {OUT_STEM}.png / .pdf")
