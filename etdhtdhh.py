import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 12))

# Define rectangle elements (Process boxes)
rect_params = [
    (0.3, 0.9, 0.4, 0.08, "Start"),
    (0.3, 0.8, 0.4, 0.1, "User uploads two signature images"),
    (0.3, 0.7, 0.4, 0.1, "Preprocess images (Grayscale, Resize, Crop)"),
    (0.3, 0.6, 0.4, 0.1, "Extract keypoints using ORB/SIFT"),
    (0.3, 0.5, 0.4, 0.1, "Compare keypoints using matching algorithm"),
    (0.3, 0.4, 0.4, 0.1, "Compare Structural Similarity (SSIM)"),
    (0.3, 0.3, 0.4, 0.1, "Compare Histogram Similarity"),
    (0.3, 0.2, 0.4, 0.1, "Calculate final similarity score"),
    (0.3, 0.1, 0.4, 0.08, "End")
]

# Add the rectangles to the plot
for x, y, width, height, label in rect_params:
    ax.add_patch(mpatches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", edgecolor="black", facecolor="#e6f7ff"))
    ax.text(x + width/2, y + height/2, label, ha="center", va="center", fontsize=12)

# Define arrows (flow between steps)
arrow_params = [
    (0.5, 0.8, 0.5, 0.74),
    (0.5, 0.7, 0.5, 0.64),
    (0.5, 0.6, 0.5, 0.54),
    (0.5, 0.5, 0.5, 0.44),
    (0.5, 0.4, 0.5, 0.34),
    (0.5, 0.3, 0.5, 0.24),
    (0.5, 0.2, 0.5, 0.12),
]

# Draw the arrows
for x_start, y_start, x_end, y_end in arrow_params:
    ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=2))

# Set plot limits and remove axis
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Show the plot
plt.tight_layout()
plt.show()
