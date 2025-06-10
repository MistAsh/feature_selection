import numpy as np
from matplotlib import pyplot as plt, animation
from typing import List
from feature_selection.schemas.history import History, FeatureSet

def visualization_of_probs_evolution(history: History,
                                     original_feature_names: List[str],
                                     original_best_subset: FeatureSet,
                                     save_path: str = None,
                                     gif: bool = False):
    """
    Animates the evolution of feature probabilities during a search process.
    This version highlights a fixed, predefined best feature set.

    Args:
        history (History): The history object containing a list of Iteration objects.
        original_feature_names (list of str): Names of all features.
        original_best_subset (FeatureSet): The specific feature set to be
                                            consistently highlighted in yellow.
        save_path (str, optional): Path to save the animation. Defaults to None.
        gif (bool): Whether to save as GIF. If False, saves as MP4. Defaults to False.
    """
    if not history:
        raise ValueError("History is empty - nothing to visualize")

    # --- CORRECTION: Use the built-in .to_mask() method ---
    # The FeatureSet object already knows how to create its own mask based on
    # the feature indices it holds and the total number of original features.
    original_best_mask = original_best_subset.to_mask()
    # --- END CORRECTION ---

    # Define labels and colors, which will be constant throughout the animation.
    feature_labels = [
        f"{name}★" if is_best else name
        for name, is_best in zip(original_feature_names, original_best_mask)
    ]
    colors = [
        'gold' if is_best else 'mediumseagreen'
        for is_best in original_best_mask
    ]

    # Initialize the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(bottom=0.3)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_xlabel('Features')

    def update(frame: int):
        """This function is called for each frame of the animation."""
        ax.clear()
        ax.set_ylim(0, 1)

        # Get data for the current frame
        current_iteration = history[frame]
        current_probs = current_iteration.metadata['probs']
        current_step = current_iteration.iteration

        # Use the pre-calculated labels and colors.
        # The underlying bar heights (probabilities) are the only thing that changes.
        ax.set_xticks(range(len(feature_labels)))
        ax.set_xticklabels(feature_labels, rotation=45, ha='right')

        bars = ax.bar(range(len(feature_labels)), current_probs,
                      color=colors, edgecolor='black')

        ax.set_title(f'Feature Selection Probabilities (Step {current_step})')
        ax.set_ylabel('Probability')
        ax.set_xlabel('Features')

        # Add probability values on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}', ha='center', va='bottom')

        return bars

    # Create and save or show the animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(history), interval=200, blit=False
    )

    if save_path:
        writer_engine = 'pillow' if gif else 'ffmpeg'
        fps = 5
        try:
            writer = animation.writers[writer_engine](fps=fps)
            print(f"Saving animation to {save_path} using {writer_engine}...")
            anim.save(save_path, writer=writer)
            print("Done.")
        except KeyError:
            print(f"Error: The '{writer_engine}' writer is not available.")
            print("Please install Pillow (for GIFs) or FFmpeg (for MP4s).")
        finally:
            plt.close(fig)
    else:
        plt.show()
        plt.close(fig)