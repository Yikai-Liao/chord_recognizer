import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from .score import chord_score_batch
from .config import CHORD_CONFIG

MAX_PREV = 5


def score_weight_from_beat(n_frame: int):
    time_index = np.arange(n_frame, dtype=np.int32)
    weight = np.zeros(n_frame, dtype=np.float32)
    weight[time_index % 2 == 0] = 0.2  # even beat
    weight[time_index % 4 == 2] += 0.15  # half downbeat
    return weight


def decode_chords(beat_chroma: np.ndarray, beat_bass: np.ndarray) -> pd.DataFrame:
    """
    对提取得到的 pitch 与 bass 的数值，进行打分，并使用动态规划解析出最佳对和弦排列

    :param beat_chroma: shape 为 [n_frame, 12]，每一拍的 pitch 特征，由 feature.extrac_chord_feature() 得到
    :param beat_bass: shape 为 [n_frame, 12]，每一拍的 bass 特征，由 feature.extrac_chord_feature() 得到
    :return: 以 pd.DataFrame 的类型，返回解析出的和弦，时间单位为 1拍
    """
    n_frame = len(beat_bass)
    # 每一beat的累计得分
    beat_chroma_pad = np.pad(beat_chroma, ((MAX_PREV - 1, 0), (0, 0)))
    beat_bass_pad = np.pad(beat_bass, ((MAX_PREV - 1, 0), (0, 0)))

    cum_chroma = np.stack([
        np.sum(  # sliding window requires np.__version__ >= 1.20
            sliding_window_view(beat_chroma_pad[MAX_PREV - j:], j, axis=0),
            axis=-1
        ) for j in range(1, MAX_PREV + 1)
    ])

    cum_bass = np.stack([
        np.sum(
            sliding_window_view(beat_bass_pad[MAX_PREV - j:], j, axis=0),
            axis=-1
        ) for j in range(1, MAX_PREV + 1)
    ])

    scores = chord_score_batch(
        cum_chroma.reshape(-1, 12),
        cum_bass.reshape(-1, 12)
    ).reshape((MAX_PREV, n_frame, -1))  # -1 for class number

    weight = score_weight_from_beat(n_frame).reshape(n_frame, 1)
    for j in range(MAX_PREV):
        scores[j, j:] += j * 0.7 + weight[: n_frame - j]

    choices = np.argmax(scores, axis=-1)
    cum_scores = np.full(n_frame, -np.inf)
    final_choices = np.zeros(n_frame, dtype=np.int32)
    start_pos = np.zeros(n_frame, dtype=np.int32)
    for i in range(n_frame):
        for j in range(MAX_PREV):
            if i - j < 0:
                break
            best_choice = choices[j, i]  # [MAX_PREV, n_frame]
            pre_score = 0 if i - j == 0 else cum_scores[i - j - 1]
            cur_score = pre_score + scores[j, i, best_choice]
            # 如果累计得分更高
            if cum_scores[i] < cur_score:
                cum_scores[i] = cur_score
                final_choices[i] = best_choice
                start_pos[i] = i - j - 1
            # if j > 0 and (i - j + 1) % 4 == 0:  # downbeat
            #     break

    result = []
    end = n_frame - 1
    chord_names = CHORD_CONFIG['name']
    chord_pitches = CHORD_CONFIG['pitch']
    while end > 0:
        start = start_pos[end] + 1
        choice = final_choices[end]
        name = chord_names[choice]
        if len(result) > 0 and result[-1][2] == name:
            result[-1][0] = start
        else:
            result.append([start, end, name, chord_pitches[choice]])
        end = start - 1

    return pd.DataFrame(result[::-1], columns=['start', 'end', 'name', 'pitch'])
