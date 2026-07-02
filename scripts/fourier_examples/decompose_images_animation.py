# fourier_synthesis_precompute.py

import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from matplotlib.animation import FuncAnimation, FFMpegWriter


def calculate_2dft(arr):
    ft = np.fft.ifftshift(arr)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)


def calculate_2dift(arr):
    ift = np.fft.ifftshift(arr)
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    return ift.real


def calculate_distance_from_centre(coords, centre):
    return np.sqrt((coords[0] - centre) ** 2 + (coords[1] - centre) ** 2)


def find_symmetric_coordinates(coords, centre):
    return (centre + (centre - coords[0]), centre + (centre - coords[1]))


# ------------------------------------------------------------------
# Load image
# ------------------------------------------------------------------

image = ski.data.camera()
# zoom in to a smaller region for faster computation
image = ski.transform.rescale(image, 0.1, anti_aliasing=True)

array_size = image.shape[0]
centre = (array_size - 1) // 2

# ------------------------------------------------------------------
# Fourier transform
# ------------------------------------------------------------------

ft = calculate_2dft(image)

# ------------------------------------------------------------------
# Generate coordinates ordered by distance from centre
# ------------------------------------------------------------------

coords_left_half = [(x, y) for x in range(array_size) for y in range(centre + 1)]

coords_left_half.sort(key=lambda c: calculate_distance_from_centre(c, centre))

# ------------------------------------------------------------------
# PRECOMPUTE EVERY FOURIER COMPONENT ONCE
# ------------------------------------------------------------------

print("Precomputing Fourier components...")

components = []

for i, coords in enumerate(coords_left_half):
    if coords[1] == centre and coords[0] > centre:
        continue

    symm_coords = find_symmetric_coordinates(coords, centre)

    tmp = np.zeros_like(ft, dtype=complex)

    tmp[coords] = ft[coords]
    tmp[symm_coords] = ft[symm_coords]

    component = calculate_2dift(tmp)

    components.append(component)

    if i % 500 == 0:
        print(f"Computed {i}/{len(coords_left_half)}")

print(f"Stored {len(components)} components")

# ------------------------------------------------------------------
# PREPARE FRAMES FOR ANIMATION
# ------------------------------------------------------------------

print("Generating sequence of frames for the video...")
frames_to_render = []
rec_image = np.zeros_like(image, dtype=float)

display_all_until = 1000
display_step = 2
next_display = display_all_until + display_step

# --- Video Timing Settings ---
FPS = 30
slow_duration_seconds = 0.5
slow_frames_count = int(FPS * slow_duration_seconds)  # 15 frames at 30fps
slow_until_component = 5  # Number of components to hold on screen

for idx, component in enumerate(components, start=1):
    rec_image += component

    if idx < display_all_until or idx == next_display:
        if idx > display_all_until:
            next_display += display_step
            display_step += 10

        if idx % 500 == 0 or idx < display_all_until:
            # Save a copy of the current state for this frame
            frame_data = (component, rec_image.copy(), idx)

            # If it's one of the first few components, duplicate the frame
            # so it stays on screen longer in the final constant-framerate video
            if idx <= slow_until_component:
                frames_to_render.extend([frame_data] * slow_frames_count)
            else:
                frames_to_render.append(frame_data)

print(f"Total frames to render: {len(frames_to_render)}")

# ------------------------------------------------------------------
# ANIMATE AND EXPORT TO MP4
# ------------------------------------------------------------------

print("Rendering video... This may take a moment.")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Initialize the axes with empty data or the first frame
im_comp = ax1.imshow(frames_to_render[0][0], cmap="gray")
ax1.axis("off")
title_comp = ax1.set_title("Fourier Component (1)")

im_rec = ax2.imshow(frames_to_render[0][1], cmap="gray")
ax2.axis("off")
title_rec = ax2.set_title("Superposition/Interference of Components")

plt.tight_layout()


def update(frame_idx):
    component, rec, idx = frames_to_render[frame_idx]

    # Update image data instead of clearing the plot (much faster)
    im_comp.set_data(component)
    im_rec.set_data(rec)

    # Dynamically update the color limits based on the new data
    im_comp.set_clim(component.min(), component.max())
    im_rec.set_clim(rec.min(), rec.max())

    # Update title
    title_comp.set_text(f"Fourier Component ({idx})")

    return im_comp, im_rec, title_comp


# Create the animation object
ani = FuncAnimation(
    fig,
    update,
    frames=len(frames_to_render),
    interval=1000 / FPS,  # fallback for live display timing
    blit=True,
)

# Set up the FFmpeg writer with our desired FPS
writer = FFMpegWriter(fps=FPS, metadata=dict(artist="Matplotlib"), bitrate=1800)

# Save to mp4
output_filename = "fourier_synthesis.mp4"
ani.save(output_filename, writer=writer)

print(f"Video saved successfully as {output_filename}!")
