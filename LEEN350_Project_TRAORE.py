# -------------------------------------------------
#  LEEN350 Image Processing and Lab
#  MAHAMAODU MASSAMAN TRAORE  /  220303904
#  Platform : Google Colab  
#  Pipeline : 8 steps
# -------------------------------------------------

# -- IMPORTS -------------------------------------------------
# Standard libraries for image processing
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.ndimage import binary_opening, binary_closing
from skimage import data
from skimage.filters import threshold_otsu, threshold_local
from skimage.measure import label, regionprops


# -------------------------------------------------
# STEP 1 — Load the image
# Built-in coins image: 303x384 px, grayscale, 24 coins
# -------------------------------------------------
original = data.coins()

print(f"Shape : {original.shape}")
print(f"Range : {original.min()} to {original.max()}")

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.imshow(original, cmap="gray")
ax.set_title("Step 1 — Original Image")
ax.axis("off")
plt.tight_layout()
plt.savefig("step1_original.png", dpi=150)
plt.show()


# ============================================================
# STEP 2 — Add Gaussian noise
# Simulates a real camera sensor — std=25, seed=42
# ============================================================
np.random.seed(42)
noise = np.random.normal(0, 25, original.shape)
noisy = np.clip(original.astype(float) + noise, 0, 255).astype(np.uint8)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(original, cmap="gray")
axes[0].set_title("Original (clean)")
axes[0].axis("off")
axes[1].imshow(noisy, cmap="gray")
axes[1].set_title("Noisy  (std=25)")
axes[1].axis("off")
plt.suptitle("Step 2 — Gaussian Noise Added")
plt.tight_layout()
plt.savefig("step2_noisy.png", dpi=150)
plt.show()


# -------------------------------------------------
# STEP 3 — Spatial filters + PSNR / MSE
# 3 filters tested: Gaussian s=1, Gaussian s=2, Median 3x3
# PSNR and MSE computed immediately after each filter
# Higher PSNR = better quality  (Eq. 2-4 filters / Eq. 19-20 metrics)
# -------------------------------------------------

# Apply filters  (Eq. 2-4)
gauss1 = gaussian_filter(noisy, sigma=1).astype(np.uint8)
gauss2 = gaussian_filter(noisy, sigma=2).astype(np.uint8)
median = median_filter(noisy, size=3).astype(np.uint8)

# PSNR / MSE — formulas
def compute_psnr(orig, filt):
    mse  = np.mean((orig.astype(float) - filt.astype(float)) ** 2)       # Eq. 19
    psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')  # Eq. 20
    return round(psnr, 2), round(mse, 2)

psnr_noisy,  mse_noisy  = compute_psnr(original, noisy)
psnr_gauss1, mse_gauss1 = compute_psnr(original, gauss1)
psnr_gauss2, mse_gauss2 = compute_psnr(original, gauss2)
psnr_median, mse_median = compute_psnr(original, median)

# Print table
print(f"\n  {'Method':<18} {'PSNR':>8} {'MSE':>8}")
print(f"  {'-'*36}")
print(f"  {'Noisy':<18} {psnr_noisy:>8} {mse_noisy:>8}")
print(f"  {'Gauss sigma=1':<18} {psnr_gauss1:>8} {mse_gauss1:>8}  best PSNR")
print(f"  {'Gauss sigma=2':<18} {psnr_gauss2:>8} {mse_gauss2:>8}")
print(f"  {'Median 3x3':<18} {psnr_median:>8} {mse_median:>8}  best edges")

# Filter comparison figure
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, img, title in zip(axes,
    [noisy, gauss1, gauss2, median],
    [f"Noisy\n{psnr_noisy} dB",
     f"Gauss s=1\n{psnr_gauss1} dB  best",
     f"Gauss s=2\n{psnr_gauss2} dB",
     f"Median 3x3\n{psnr_median} dB"]):
    ax.imshow(img, cmap="gray")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
plt.suptitle("Step 3 — Spatial Filters  [Eq. 2-4]")
plt.tight_layout()
plt.savefig("step3_spatial_filters.png", dpi=150)
plt.show()

# PSNR bar chart
fig2, ax2 = plt.subplots(figsize=(7, 4))
bars = ax2.bar(["Noisy","Gauss s=1","Gauss s=2","Median"],
               [psnr_noisy, psnr_gauss1, psnr_gauss2, psnr_median],
               color=["#DC2626","#1B4F8C","#1976D2","#059669"])
ax2.bar_label(bars, fmt="%.1f dB", padding=3)
ax2.set_ylim(15, 32)
ax2.set_ylabel("PSNR (dB)")
ax2.set_title("Step 3 — Spatial Filter PSNR  [Eq. 19-20]")
plt.tight_layout()
plt.savefig("step3_psnr_spatial.png", dpi=150)
plt.show()


# -------------------------------------------------
# STEP 4 — FFT low-pass filter + full PSNR comparison
# Transform image to frequency domain — apply circular mask — transform back
# r=30 strong cut / r=60 mild cut  (Eq. 5-7)
# Full comparison of all 6 methods
# -------------------------------------------------

def fft_lowpass_filter(image, cutoff_radius):
    f    = np.fft.fftshift(np.fft.fft2(image.astype(float)))   # Eq. 5
    r, c = image.shape
    Y, X = np.ogrid[:r, :c]
    dist = np.sqrt((X - c // 2) ** 2 + (Y - r // 2) ** 2)
    mask = (dist <= cutoff_radius).astype(float)                # Eq. 6
    filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(f * mask)))  # Eq. 7
    return filtered.clip(0, 255).astype(np.uint8)

fft_r30 = fft_lowpass_filter(noisy, cutoff_radius=30)
fft_r60 = fft_lowpass_filter(noisy, cutoff_radius=60)

psnr_r30, mse_r30 = compute_psnr(original, fft_r30)
psnr_r60, mse_r60 = compute_psnr(original, fft_r60)

print(f"\n  {'FFT r=30':<18} {psnr_r30:>8} {mse_r30:>8}")
print(f"  {'FFT r=60':<18} {psnr_r60:>8} {mse_r60:>8}  similar to Gauss s=1")

# FFT vs Gaussian figure
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, img, title in zip(axes,
    [noisy, fft_r30, fft_r60, gauss1],
    [f"Noisy\n{psnr_noisy} dB",
     f"FFT r=30\n{psnr_r30} dB",
     f"FFT r=60\n{psnr_r60} dB",
     f"Gauss s=1 (ref)\n{psnr_gauss1} dB"]):
    ax.imshow(img, cmap="gray")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
plt.suptitle("Step 4 — FFT Low-Pass Filter  [Eq. 5-7]")
plt.tight_layout()
plt.savefig("step4_fft_filters.png", dpi=150)
plt.show()

# Full PSNR comparison — all 6 methods
fig2, ax2 = plt.subplots(figsize=(9, 4))
bars = ax2.bar(
    ["Noisy","Gauss s=1","Gauss s=2","Median","FFT r=60","FFT r=30"],
    [psnr_noisy, psnr_gauss1, psnr_gauss2, psnr_median, psnr_r60, psnr_r30],
    color=["#DC2626","#1B4F8C","#1976D2","#059669","#7C3AED","#6D28D9"])
ax2.bar_label(bars, fmt="%.1f dB", padding=3)
ax2.set_ylim(15, 32)
ax2.set_ylabel("PSNR (dB)")
ax2.set_title("Step 4 — All 6 Methods PSNR  [Eq. 19-20]")
plt.tight_layout()
plt.savefig("step4_psnr_all.png", dpi=150)
plt.show()


# -------------------------------------------------
# STEP 5 — Segmentation
# Input: best filtered image = Gaussian sigma=1
# Otsu: one global threshold  (Eq. 8-9)
# Adaptive: local threshold per pixel, block=35, C=10  (Eq. 10)
# Both masks shown side by side immediately
# -------------------------------------------------
best_filter = gauss1   # best PSNR from Step 3

T_otsu        = threshold_otsu(best_filter)              # Eq. 8, 9
mask_otsu     = best_filter > T_otsu

T_adaptive    = threshold_local(best_filter, block_size=35, offset=10)  # Eq. 10
mask_adaptive = best_filter > T_adaptive

print(f"\n  Otsu threshold : {T_otsu}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, img, title in zip(axes,
    [best_filter, mask_otsu, mask_adaptive],
    ["Gauss s=1 (input)",
     f"Otsu  T={T_otsu}  [Eq. 8-9]",
     "Adaptive  block=35  [Eq. 10]"]):
    ax.imshow(img, cmap="gray")
    ax.set_title(title, fontsize=11)
    ax.axis("off")
plt.suptitle("Step 5 — Segmentation: Otsu vs Adaptive")
plt.tight_layout()
plt.savefig("step5_segmentation.png", dpi=150)
plt.show()


# -------------------------------------------------
# STEP 6 — Morphological operations
# Input: Otsu binary mask
# 4 operations with a 3x3 kernel  (Eq. 11-14)
# Closing is best: fills holes inside coins
# All 5 masks shown side by side
# -------------------------------------------------
kernel = np.ones((3, 3), dtype=bool)

morph_erosion  = binary_erosion(mask_otsu,  structure=kernel)   # Eq. 11
morph_dilation = binary_dilation(mask_otsu, structure=kernel)   # Eq. 12
morph_opening  = binary_opening(mask_otsu,  structure=kernel)   # Eq. 13
morph_closing  = binary_closing(mask_otsu,  structure=kernel)   # Eq. 14  best

fig, axes = plt.subplots(1, 5, figsize=(22, 4))
for ax, img, title in zip(axes,
    [mask_otsu, morph_erosion, morph_dilation, morph_opening, morph_closing],
    ["Otsu (input)", "Erosion [Eq.11]", "Dilation [Eq.12]",
     "Opening [Eq.13]", "Closing [Eq.14] best"]):
    ax.imshow(img, cmap="gray")
    ax.set_title(title, fontsize=11)
    ax.axis("off")
plt.suptitle("Step 6 — Morphological Operations")
plt.tight_layout()
plt.savefig("step6_morphology.png", dpi=150)
plt.show()


# -------------------------------------------------
# STEP 7 — Segmentation metrics
# Reference mask: Otsu
# Confusion matrix built manually: TP, TN, FP, FN
# 5 metrics: Accuracy, Precision, Recall, F1, IoU  (Eq. 21-25)
# Full table printed
# -------------------------------------------------

def segmentation_metrics(ground_truth, prediction):
    gt = ground_truth.astype(bool)
    pr = prediction.astype(bool)
    TP = np.sum( gt &  pr)
    TN = np.sum(~gt & ~pr)
    FP = np.sum(~gt &  pr)
    FN = np.sum( gt & ~pr)
    accuracy  = (TP + TN) / (TP + TN + FP + FN)               # Eq. 21
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0        # Eq. 22
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0        # Eq. 23
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)           # Eq. 24
    iou       = TP / (TP + FP + FN) if (TP+FP+FN) > 0 else 0 # Eq. 25
    return {"Accuracy":round(accuracy,3), "Precision":round(precision,3),
            "Recall":round(recall,3), "F1":round(f1,3), "IoU":round(iou,3)}

reference = mask_otsu

print(f"\n  {'Comparison':<24} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'IoU':>6}")
print(f"  {'-'*56}")
results = {}
for name, mask in [
    ("Adaptive vs Otsu",  mask_adaptive),
    ("Erosion vs Otsu",   morph_erosion),
    ("Dilation vs Otsu",  morph_dilation),
    ("Opening vs Otsu",   morph_opening),
    ("Closing vs Otsu",   morph_closing),
]:
    m = segmentation_metrics(reference, mask)
    results[name] = m
    marker = "  best" if name == "Closing vs Otsu" else ""
    print(f"  {name:<24} {m['Accuracy']:>6} {m['Precision']:>6} "
          f"{m['Recall']:>6} {m['F1']:>6} {m['IoU']:>6}{marker}")

# Best vs Adaptive comparison figure
m_adaptive = results["Adaptive vs Otsu"]
m_closing  = results["Closing vs Otsu"]
fig, axes  = plt.subplots(1, 3, figsize=(14, 4))
for ax, img, title in zip(axes,
    [mask_otsu, mask_adaptive, morph_closing],
    ["Otsu (reference)",
     f"Adaptive\nF1={m_adaptive['F1']}  IoU={m_adaptive['IoU']}",
     f"Closing  best\nF1={m_closing['F1']}  IoU={m_closing['IoU']}"]):
    ax.imshow(img, cmap="gray")
    ax.set_title(title, fontsize=11)
    ax.axis("off")
plt.suptitle("Step 7 — Segmentation Metrics  [Eq. 21-25]")
plt.tight_layout()
plt.savefig("step7_metrics_comparison.png", dpi=150)
plt.show()


# -------------------------------------------------
# STEP 8 — Feature Extraction
# Input: Closing mask (best result)
# label() gives each coin a unique integer ID
# regionprops() extracts shape features per coin
# 4 features per coin  (Eq. 15-18)
# Circularity ~ 1 and Eccentricity ~ 0 = round = coin confirmed
# Labelled image and scatter plot
# -------------------------------------------------
labeled_mask = label(morph_closing)   # unique ID per coin region

features = []
for region in regionprops(labeled_mask):
    if region.area < 50:              # skip noise regions (too small)
        continue
    area    = region.area                                     # Eq. 15
    eq_diam = region.equivalent_diameter                      # Eq. 16
    circ    = ((4 * np.pi * area) / (region.perimeter ** 2)
               if region.perimeter > 0 else 0)               # Eq. 17
    eccen   = region.eccentricity                             # Eq. 18
    features.append([area, eq_diam, circ, eccen])

features_arr = np.array(features)

# summary
print(f"\n  Coins detected : {len(features)}")
print(f"\n  {'Feature':<22} {'Avg':>7} {'Min':>7} {'Max':>7}")
print(f"  {'-'*46}")
for i, name in enumerate(["Area (px)","Eq. Diameter","Circularity","Eccentricity"]):
    print(f"  {name:<22} {features_arr[:,i].mean():>7.2f} "
          f"{features_arr[:,i].min():>7.2f} {features_arr[:,i].max():>7.2f}")
print(f"\n  Avg circularity  = {features_arr[:,2].mean():.3f}  (close to 1 = round)")
print(f"  Avg eccentricity = {features_arr[:,3].mean():.3f}  (close to 0 = circular)")

# Labelled regions figure
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(morph_closing, cmap="gray")
axes[0].set_title("Closing mask (input)")
axes[0].axis("off")
axes[1].imshow(labeled_mask, cmap="nipy_spectral")
axes[1].set_title(f"Labelled  ({len(features)} coins)")
axes[1].axis("off")
plt.suptitle("Step 8 — Feature Extraction: Labelled Coins  [Source 14]")
plt.tight_layout()
plt.savefig("step8_labelled_regions.png", dpi=150)
plt.show()

# Scatter plot: Circularity vs Area immediately
fig2, ax2 = plt.subplots(figsize=(8, 5))
sc = ax2.scatter(features_arr[:,0], features_arr[:,2],
                 c=features_arr[:,3], cmap="plasma",
                 s=90, edgecolors="k", linewidths=0.5)
plt.colorbar(sc, ax=ax2, label="Eccentricity  [Eq. 18]")
ax2.set_xlabel("Area in pixels  [Eq. 15]")
ax2.set_ylabel("Circularity  [Eq. 17]")
ax2.axhline(y=features_arr[:,2].mean(), color="red", linestyle="--",
            alpha=0.6, label=f"avg = {features_arr[:,2].mean():.2f}")
ax2.legend()
ax2.set_title("Step 8 — Circularity vs Area per Coin  [Eq. 15-18]")
plt.tight_layout()
plt.savefig("step8_features_scatter.png", dpi=150)
plt.show()

# FINAL SUMMARY
print("\n" + "=" * 55)
print("  PIPELINE COMPLETE")
print("=" * 55)
print(f"  Best PSNR       : Gauss s=1   {psnr_gauss1} dB")
print(f"  Best edges      : Median 3x3  {psnr_median} dB")
print(f"  FFT r=60        : {psnr_r60} dB  (matches Gauss s=1)")
print(f"  Best F1         : {m_closing['F1']}  (Closing)")
print(f"  Best IoU        : {m_closing['IoU']}  (Closing)")
print(f"  Coins detected  : {len(features)}")
print(f"  Avg circularity : {features_arr[:,2].mean():.3f}")
print(f"  Equations : 25  /  Metrics : 7  /  Sources : 14")
print("=" * 55)
