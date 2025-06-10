from typing import List, Any

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from feature_selection.schemas.history import History


def extract_all_points_from_history(
        history: History
) -> tuple[List[float], List[float]]:
    x_coords = []
    y_coords = []
    for it in history:
        for candidate in it.candidate_features:
            x_coords.append(len(candidate.features))
            y_coords.append(candidate.score)
    return x_coords, y_coords


def calculate_upper_envelope(x_coords, y_coords):
    """Рассчитывает верхнюю огибающую для набора точек, игнорируя -np.inf в y_coords."""
    if not isinstance(x_coords, (list, np.ndarray, pd.Series)) or \
            not isinstance(y_coords, (list, np.ndarray, pd.Series)):
        return [], []

    x_coords_arr = np.asarray(x_coords)
    y_coords_arr = np.asarray(y_coords)

    if len(x_coords_arr) == 0 or len(y_coords_arr) == 0:
        return [], []
    if len(x_coords_arr) != len(y_coords_arr):
        min_len = min(len(x_coords_arr), len(y_coords_arr))
        x_coords_arr = x_coords_arr[:min_len]
        y_coords_arr = y_coords_arr[:min_len]
        print(
            f"Warning: x_coords and y_coords have different lengths. Truncated to {min_len}.")

    # Фильтруем NaN из X и Y, а также Inf из Y (X должен быть конечным)
    # Только конечные Y могут быть частью верхней огибающей
    finite_x_indices = np.isfinite(x_coords_arr)
    finite_y_indices = np.isfinite(
        y_coords_arr)  # Отфильтровываем и NaN, и Inf/-Inf для Y
    valid_indices = finite_x_indices & finite_y_indices

    x_coords_filtered = x_coords_arr[valid_indices]
    y_coords_filtered = y_coords_arr[valid_indices]

    if len(x_coords_filtered) == 0:
        return [], []

    df = pd.DataFrame(
        {'num_features': x_coords_filtered, 'score': y_coords_filtered})
    df = df.drop_duplicates(subset=['num_features', 'score'])

    if df.empty:
        return [], []

    try:
        idx = df.groupby('num_features')['score'].idxmax()
        envelope_df = df.loc[idx]
    except (KeyError, ValueError):
        return [], []

    envelope_df = envelope_df.sort_values(by='num_features').reset_index(
        drop=True)
    return envelope_df['num_features'].tolist(), envelope_df['score'].tolist()


def animate_feature_selection_progress(
        history_data: History,
        feature_names: List[str],
        total_num_features: int,
        name: str,
        all_points_x: List[float] | None = None,
        all_points_y: List[float] | None = None,
        envelope_line_x: List[float] | None = None,
        envelope_line_y: List[float] | None = None,
        auto_calculate_envelope: bool = False,
        save_path: str | None = None,
        gif: bool = False,
        interval: int = 1000,
        fps: int = 5
):
    if not history_data or not history_data.iterations:
        print(
            "The selection history (History) is empty or contains "
            "no iterations, there is nothing to animate.")
        return

    animation_data = []
    history_iterations_map = {it.iteration: it for it in
                              history_data.iterations}

    for iteration_obj in history_data.iterations:
        if iteration_obj.selected_features:
            selected_fs = iteration_obj.selected_features
            num_selected_features = len(selected_fs.features)
            score = selected_fs.score
            selected_indices = selected_fs.to_list()

            animation_data.append({
                'num_features': num_selected_features,
                'score': score,
                'feature_indices': selected_indices,
                'iter': iteration_obj.iteration
            })

    if not animation_data:
        print(
            "No data to animate after "
            "History processing (selected_features not found in iterations).")
        return

    all_points_x_to_plot: List[float] | None = None
    all_points_y_to_plot: List[float] | None = None
    x_for_envelope_calc: List[float] | None = None
    y_for_envelope_calc: List[float] | None = None

    if all_points_x is not None and all_points_y is not None:
        all_points_x_to_plot = all_points_x
        all_points_y_to_plot = all_points_y
        x_for_envelope_calc = all_points_x
        y_for_envelope_calc = all_points_y
    else:
        collected_all_x: List[float] = []
        collected_all_y: List[float] = []
        for iteration_obj in history_data.iterations:
            if iteration_obj.selected_features:
                fs = iteration_obj.selected_features
                if not np.isnan(len(fs.features)):
                    collected_all_x.append(float(len(fs.features)))
                    collected_all_y.append(fs.score)
            for fs in iteration_obj.candidate_features:
                if not np.isnan(len(fs.features)):
                    collected_all_x.append(float(len(fs.features)))
                    collected_all_y.append(fs.score)

        if collected_all_x and collected_all_y:
            all_points_x_to_plot = collected_all_x
            all_points_y_to_plot = collected_all_y
            x_for_envelope_calc = collected_all_x
            y_for_envelope_calc = collected_all_y

    actual_envelope_x, actual_envelope_y = envelope_line_x, envelope_line_y
    if auto_calculate_envelope and (
            envelope_line_x is None or envelope_line_y is None):
        if x_for_envelope_calc is not None and y_for_envelope_calc is not None and \
                len(x_for_envelope_calc) > 0 and len(y_for_envelope_calc) > 0:
            actual_envelope_x, actual_envelope_y = calculate_upper_envelope(
                x_for_envelope_calc,
                y_for_envelope_calc)
        else:
            if not (x_for_envelope_calc and y_for_envelope_calc and \
                    len(x_for_envelope_calc) > 0 and len(
                        y_for_envelope_calc) > 0):
                print("Unable to calculate envelope: not enough data "
                      "(all points or History).")
            actual_envelope_x, actual_envelope_y = [], []

    fig, ax = plt.subplots(figsize=(14, 10))
    plt.subplots_adjust(right=0.78, top=0.9, left=0.1)

    plot_xs_all_types: List[float] = []
    plot_ys_all_types: List[float] = []

    def extend_if_not_none(target_list: List[float], source_data: Any):
        if source_data is not None:
            source_arr = np.asarray(source_data)
            if target_list is plot_xs_all_types:
                target_list.extend(source_arr[~np.isnan(source_arr)])
            else:
                target_list.extend(source_arr[~np.isnan(source_arr)])

    for d in animation_data:
        if not np.isnan(d['num_features']): plot_xs_all_types.append(
            d['num_features'])
        if not np.isnan(d['score']): plot_ys_all_types.append(d['score'])

    extend_if_not_none(plot_xs_all_types, all_points_x_to_plot)
    extend_if_not_none(plot_ys_all_types, all_points_y_to_plot)
    if actual_envelope_x is not None: extend_if_not_none(plot_xs_all_types,
                                                         actual_envelope_x)
    if actual_envelope_y is not None: extend_if_not_none(plot_ys_all_types,
                                                         actual_envelope_y)

    finite_plot_xs = [x for x in plot_xs_all_types if np.isfinite(x)]
    finite_plot_ys = [y for y in plot_ys_all_types if np.isfinite(y)]

    if not finite_plot_xs:
        min_x_val, max_x_val = 0, (
            total_num_features if total_num_features > 0 else 1)
    else:
        min_x_val = min(finite_plot_xs)
        max_x_val = max(finite_plot_xs)

    if not finite_plot_ys:
        min_y_val, max_y_val = 0, 1
    else:
        min_y_val = min(finite_plot_ys)
        max_y_val = max(finite_plot_ys)

    min_x_ax, max_x_ax = (min_x_val - 0.5, max_x_val + 0.5)
    min_y_ax, max_y_ax = (
        min_y_val - 0.05 * abs(min_y_val) if min_y_val != 0 else -0.05,
        max_y_val + 0.05 * abs(max_y_val) if max_y_val != 0 else 0.05)

    if min_x_ax >= max_x_ax - 1: max_x_ax = min_x_ax + 1
    if min_y_ax >= max_y_ax - 0.01: max_y_ax = min_y_ax + 0.1
    if total_num_features > 0 and max_x_ax > total_num_features + 0.5: max_x_ax = total_num_features + 0.5
    if min_x_ax < -0.5: min_x_ax = -0.5

    if min_y_val >= 0 and min_y_ax > 0: min_y_ax = -0.01
    if max_y_val <= 1 and max_y_ax < 1: max_y_ax = 1.01

    def _update_best_point_logic(num_f, s, current_best_s, current_best_nf,
                                 found_flag_ref_list):
        if np.isnan(s) or np.isnan(num_f):
            return current_best_s, current_best_nf, found_flag_ref_list
        is_better = False
        if not found_flag_ref_list[0]:
            if np.isfinite(s):
                is_better = True
        elif s > current_best_s:
            is_better = True
        elif s == current_best_s:
            if num_f < current_best_nf:
                is_better = True
        if is_better:
            return s, num_f, [True]
        return current_best_s, current_best_nf, found_flag_ref_list

    features_text_object = fig.text(0.80, 0.88, "",
                                    fontsize=10,
                                    verticalalignment='top',
                                    bbox=dict(boxstyle='round,pad=0.5',
                                              fc='wheat', alpha=0.8))

    def update(frame_idx):
        ax.clear()
        current_anim_data_point = animation_data[frame_idx]
        num_f_curr = current_anim_data_point['num_features']
        score_curr = current_anim_data_point['score']
        iter_num_from_anim_data = current_anim_data_point['iter']

        ax.set_xlim(min_x_ax, max_x_ax)
        ax.set_ylim(min_y_ax, max_y_ax)
        ax.set_xlabel("Number of selected features", fontsize=14)
        ax.set_ylabel("Score", fontsize=14)
        fig.suptitle(
            f"{name}: (Frame {frame_idx + 1}/{len(animation_data)})",
            fontsize=16)
        ax.grid(True, linestyle=':', alpha=0.7)

        best_score_so_far = -float('inf')
        best_num_features_so_far = -1
        found_best_point_so_far_ref = [False]

        for point_idx in range(frame_idx + 1):
            point = animation_data[point_idx]
            best_score_so_far, best_num_features_so_far, found_best_point_so_far_ref = _update_best_point_logic(
                point['num_features'], point['score'],
                best_score_so_far, best_num_features_so_far,
                found_best_point_so_far_ref
            )
        if all_points_x_to_plot is not None and all_points_y_to_plot is not None:
            _all_x_arr_bg = np.asarray(all_points_x_to_plot)
            _all_y_arr_bg = np.asarray(all_points_y_to_plot)
            for x_val, y_val in zip(_all_x_arr_bg, _all_y_arr_bg):
                best_score_so_far, best_num_features_so_far, found_best_point_so_far_ref = _update_best_point_logic(
                    x_val, y_val, best_score_so_far, best_num_features_so_far,
                    found_best_point_so_far_ref
                )
        if actual_envelope_x is not None and actual_envelope_y is not None:
            _env_x_arr_best = np.asarray(actual_envelope_x)
            _env_y_arr_best = np.asarray(actual_envelope_y)
            for x_val, y_val in zip(_env_x_arr_best, _env_y_arr_best):
                best_score_so_far, best_num_features_so_far, found_best_point_so_far_ref = _update_best_point_logic(
                    x_val, y_val, best_score_so_far, best_num_features_so_far,
                    found_best_point_so_far_ref
                )

        if all_points_x_to_plot is not None and all_points_y_to_plot is not None:
            _all_x_plot = np.asarray(all_points_x_to_plot)
            _all_y_plot = np.asarray(all_points_y_to_plot)
            valid_bg_indices = np.isfinite(_all_x_plot) & np.isfinite(
                _all_y_plot)
            ax.scatter(_all_x_plot[valid_bg_indices],
                       _all_y_plot[valid_bg_indices],
                       s=50, color='lightgrey', alpha=0.5,
                       label="All combinations considered")

        current_full_iteration_obj = history_iterations_map.get(
            iter_num_from_anim_data)
        if current_full_iteration_obj and current_full_iteration_obj.candidate_features:
            candidate_x_coords = []
            candidate_y_coords = []
            for fs in current_full_iteration_obj.candidate_features:
                if not np.isnan(len(fs.features)) and np.isfinite(fs.score):
                    candidate_x_coords.append(float(len(fs.features)))
                    candidate_y_coords.append(fs.score)

            if candidate_x_coords:
                ax.scatter(candidate_x_coords, candidate_y_coords,
                           s=35, color='cyan', alpha=0.6, marker='D',
                           label="Candidates for this iteration")

        if actual_envelope_x is not None and actual_envelope_y is not None and \
                len(actual_envelope_x) > 0 and len(actual_envelope_y) > 0:
            _env_x_plot = np.asarray(actual_envelope_x)
            _env_y_plot = np.asarray(actual_envelope_y)
            valid_env_indices = np.isfinite(_env_x_plot) & np.isfinite(
                _env_y_plot)
            if np.any(valid_env_indices):
                ax.plot(_env_x_plot[valid_env_indices],
                        _env_y_plot[valid_env_indices],
                        color='salmon', linestyle='--', linewidth=2, alpha=0.9,
                        label="Upper envelope")
                ax.scatter(_env_x_plot[valid_env_indices],
                           _env_y_plot[valid_env_indices],
                           color='red', s=70, alpha=0.7, marker='o',
                           label="Envelope points")

        path_x_arr = np.asarray(
            [d['num_features'] for d in animation_data[:frame_idx + 1]])
        path_y_arr = np.asarray(
            [d['score'] for d in animation_data[:frame_idx + 1]])
        valid_path_indices = np.isfinite(path_x_arr) & np.isfinite(path_y_arr)
        if np.any(valid_path_indices):
            ax.plot(path_x_arr[valid_path_indices],
                    path_y_arr[valid_path_indices],
                    color='royalblue', linestyle='-', marker='o',
                    alpha=0.8, markersize=8, linewidth=2.5,
                    label="Current selection path")

        if np.isfinite(num_f_curr) and np.isfinite(score_curr):
            ax.scatter([num_f_curr], [score_curr], s=180, color='blue',
                       edgecolor='black', zorder=9, linewidth=1.5,
                       label="Current point selection")

        if found_best_point_so_far_ref[0] and \
                np.isfinite(best_num_features_so_far) and \
                np.isfinite(best_score_so_far):
            ax.scatter([best_num_features_so_far], [best_score_so_far], s=450,
                       color='gold', marker='*',
                       edgecolor='black', zorder=10,
                       label="Best score")

        selected_names = []
        for i in current_anim_data_point['feature_indices']:
            if 0 <= i < len(feature_names):
                selected_names.append(feature_names[i])
            else:
                selected_names.append(f"F_{i}")

        if selected_names:
            display_features_text = "Selected features:\n" + "\n".join(
                selected_names)
        else:
            display_features_text = "Selected features:\n(none)"

        features_text_object.set_text(display_features_text)

        score_text = (f"Iteration: {iter_num_from_anim_data}\n"
                      f"Features: {num_f_curr}\n"
                      f"Score: {score_curr:.4f}" if np.isfinite(
            score_curr) else f"Score: {score_curr}")
        ax.text(0.98, 0.98, score_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.4', fc='lightblue', alpha=0.8))

        handles, labels = ax.get_legend_handles_labels()
        by_label = {}
        for handle, label in zip(handles, labels):
            if label not in by_label:
                by_label[label] = handle
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc='lower left',
                      fontsize=10)

    anim = animation.FuncAnimation(fig, update, frames=len(animation_data),
                                   interval=interval, blit=False, repeat=False)

    if save_path:
        try:
            WriterClass = animation.writers['pillow'] if gif else \
                animation.writers['ffmpeg']
            writer_instance = WriterClass(fps=fps)
            actual_save_path = save_path
            if gif and not save_path.lower().endswith('.gif'):
                actual_save_path += '.gif'
            elif not gif and not save_path.lower().endswith(
                    ('.mp4', '.avi', '.mov', '.mkv')):
                actual_save_path += '.mp4'

            print(f"Saving animation to:{actual_save_path}...")
            anim.save(actual_save_path, writer=writer_instance)
            print(f"Animation saved successfully.")
        except Exception as e:
            print(f"Error saving animation:{e}")
            print(
                "Make sure Pillow (for GIF) or FFmpeg (for MP4/other videos) "
                "are installed and available.")
            if plt.get_fignums(): plt.show()
    else:
        plt.show()

    plt.close(fig)