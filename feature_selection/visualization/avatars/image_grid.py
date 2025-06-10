import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import os

from feature_selection.visualization.avatars.constants import *
from feature_selection.visualization.avatars.encoding import (
    feature_count,
    encode_feature,
    get_mapping_values,
    convert_to_avatar,
    from_avatar_to_image,
    convert_vectors_to_image
)
from feature_selection.schemas.history import History
import matplotlib.patches as patches


def prepare_history_for_visualization(history: History):
    formatted_data = []
    for it in history:
        populations_vectors = list(
            map(lambda x: x.to_mask(), it.candidate_features)
        )
        images = convert_vectors_to_image(populations_vectors)
        title = [f'iteration: {it.iteration}']
        for key, value in it.metadata.items():
            sanitized_name = key.replace('_', ' ')
            if isinstance(value, float):
                value = f'{value:.2f}'
            title.append(f'{sanitized_name}: {value}')
        title = '\n'.join(title)
        formatted_data.append({
            'images': images,
            'title': title,
        })
    return formatted_data


def create_image_grid(images, max_size_fig=6):
    """Creates a grid of images, returns fig, axes, and image objects."""
    n_images = len(images)
    if n_images == 0:
        raise ValueError("No images to display")

    n_cols = int(np.ceil(np.sqrt(n_images)))
    n_rows = int(np.ceil(n_images / n_cols))
    max_dim = max(n_cols, n_rows)
    size = max_size_fig / max_dim

    fig, axes = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(n_cols * size, n_rows * size)
    axes_flat = axes.ravel() if n_cols * n_rows > 1 else [axes]

    image_objects = []  # Initialize as an empty list
    for ax in axes_flat:
        ax.axis('off')

    return fig, axes_flat, image_objects


def animate_grid(data, fps=1, gif=True, save_path=None, max_fig_size=6):
    """Animates a grid of images with correct title and spacing."""

    first_frame_images = data[0]['images']
    fig, axes, image_objects = create_image_grid(
        first_frame_images,
        max_fig_size
    )

    # Initial title setup
    title_text = fig.suptitle(data[0]['title'])
    title_text.set_bbox({"facecolor": "lightgray", "alpha": 0.5, "pad": 5})

    # --- Apply tight_layout *ONCE* before the animation ---
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for title

    def update(frame):
        d = data[frame]
        images = d['images']

        # Clear all axes
        for ax in axes:
            ax.cla()
            ax.axis('off')

        # Repopulate image_objects
        image_objects.clear()
        for i, img in enumerate(images):
            if i < len(axes):
                im = axes[i].imshow(img)
                image_objects.append(im)
                axes[i].axis('off')

        # Update title text
        title_text.set_text(d['title'])
        if title_text.get_text() != '':
            title_text.set_bbox(
                {"facecolor": "lightgray", "alpha": 0.5, "pad": 5})

        return []  # No blitting

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(data),
        blit=False,
        interval=10
    )
    writer = animation.PillowWriter(fps=fps) \
        if gif else animation.FFMpegWriter(fps=fps)

    if save_path is None:
        plt.show()
    else:
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        anim.save(save_path, writer=writer)
    plt.close(fig)


def visualization(
        vector: np.ndarray,
        original_names: list,
        max_size_fig: int = 12,
        dpi=100
):
    if vector.shape[0] < TOTAL_BITS_COUNT:
        vector = np.pad(
            vector,
            (0, TOTAL_BITS_COUNT - vector.shape[0])
        )
    max_length_original_name = np.max([len(name) for name in original_names])

    # Calculate layout parameters
    colum_count = len(original_names)
    bits_count_per_feature = [
        (len(values) - 1).bit_length() for values in FEATURES_MAPPING.values()
    ]
    feature_num = feature_count(colum_count)
    if sum(bits_count_per_feature[:feature_num]) < len(original_names):
        feature_num += 1
    row_count = feature_num
    margin_features = np.ceil(
        MAX_LENGTH_ENCODING_LABEL / 5)  # Width of feature description column
    margin_original_names = np.ceil(
        max_length_original_name / 5)  # Height of original names row
    title_margin = 1  # Height of title row

    # Get the number of bits needed for each feature

    # Create the figure and axis

    # Set plot limits
    x_lim = margin_features + colum_count + row_count
    y_lim = margin_original_names + row_count + title_margin
    if x_lim >= y_lim:
        x_s = max_size_fig
        y_s = max_size_fig * y_lim / x_lim
    else:
        x_s = max_size_fig * x_lim / y_lim
        y_s = max_size_fig

    fig, ax = plt.subplots(figsize=(x_s, y_s), dpi=dpi)
    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)

    # Remove axis ticks and spines for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Encode the input vector and get feature values
    encoded_vector = encode_feature(vector)
    feature_mapping = get_mapping_values(encoded_vector)
    feature_values = feature_mapping['_feature_names']

    # Display feature names and their values
    for index, feature in enumerate(FEATURES_MAPPING.keys()):
        if index >= feature_num:
            break

        x = 0
        y = index + margin_original_names
        w = margin_features
        h = 1

        # Add background for feature rows for better readability
        bg_color = "#f8f8f8" if index % 2 == 0 else "#ffffff"
        ax.add_patch(
            patches.Rectangle(
                (x, y), w, h,
                fill=True,
                facecolor=bg_color,
                edgecolor="#e0e0e0",
                lw=0.5,
            )
        )

        # Add feature name and its corresponding value
        ax.text(
            x + w / 2, y + h / 2,
            f'{feature} → {feature_values[feature]}',
            ha="center",
            va="center",
            fontsize=11,
            fontweight='bold'
        )

    # Generate and display the avatar image
    avatar = convert_to_avatar(encoded_vector)
    avatar_img = from_avatar_to_image(avatar)

    # Position for avatar image
    x1 = bit_matrix_length = len(original_names)
    x1 += margin_features
    y1 = margin_original_names
    x2 = x1 + row_count
    y2 = y1 + row_count
    extent = np.array([x1, x2, y1, y2], dtype=np.float32)

    # Add section titles
    ax.text(
        margin_features / 2,
        margin_original_names + row_count + title_margin / 2,
        "FEATURES",
        ha="center", va="center", fontsize=14, fontweight='bold'
    )
    ax.text(
        colum_count / 2 + margin_features,
        margin_original_names + row_count + title_margin / 2,
        "BINARY VECTOR",
        ha="center", va="center", fontsize=14, fontweight='bold'
    )
    ax.text(
        row_count / 2 + margin_features + colum_count,
        margin_original_names + row_count + title_margin / 2,
        "AVATAR",
        ha="center", va="center", fontsize=14, fontweight='bold'
    )

    # Display the avatar image
    ax.imshow(avatar_img, extent=extent)

    # Add border around avatar for better visual separation
    ax.add_patch(
        patches.Rectangle(
            (x1, y1),
            row_count,
            row_count,
            fill=False,
            edgecolor='gray',
            lw=1,
            linestyle='-'
        )
    )

    # Display the binary vector matrix with visual representation
    current_bit = 0
    start_pos = 0

    for i in range(row_count):
        for j in range(colum_count):
            if j >= start_pos and j - start_pos < bits_count_per_feature[i]:
                # Get the bit value
                bit = vector[current_bit]
                # Choose color based on bit value
                color = "#4285F4" if bit else "#EA4335"  # Blue for 1, Red for 0

                x = margin_features + j
                y = i + margin_original_names
                w = 1
                h = 1

                # Draw the cell
                ax.add_patch(
                    patches.Rectangle(
                        (x, y), w, h,
                        fill=True,
                        facecolor=color,
                        edgecolor='white',
                        lw=1
                    )
                )

                # Add the bit value
                ax.text(
                    x + w / 2, y + h / 2,
                    str(bit),
                    ha="center", va="center",
                    fontsize=10, color="white",
                    fontweight='bold'
                )

                current_bit += 1

        # Update starting position for next feature
        start_pos += bits_count_per_feature[i]

    # Add original feature names at the bottom
    for i, name in enumerate(original_names):
        x = margin_features + i
        y = 0
        w = 1
        h = margin_original_names

        # Alternate background colors for better readability
        bg_color = "#f0f0f0" if i % 2 == 0 else "#e8e8e8"
        ax.add_patch(
            patches.Rectangle(
                (x, y), w, h,
                fill=True,
                facecolor=bg_color,
                edgecolor='white',
                lw=0.5
            )
        )

        # Add the feature name
        ax.text(
            x + w / 2, y + h / 2,
            name,
            rotation=90,
            fontsize=9,
            ha='center',
            va='center'
        )

    # Add a title and description
    plt.suptitle(
        "Avatar Feature Encoding Visualization",
        fontsize=16,
        y=0.98
    )
    plt.figtext(
        0.5, 0.01,
        "This visualization shows how binary "
        "vectors map to avatar features",
        ha="center", fontsize=10, style='italic'
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
    plt.close(fig)
