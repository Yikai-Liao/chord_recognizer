from typing import List

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from .score import chord_score
from .config import CHORD_CONFIG
from numba import njit
from .util import TimeSignature

MAX_PREV = 8


def downbeat_and_score_weight(n_frame: int, time_signatures: List[TimeSignature]):
    if len(time_signatures) == 0:  # 默认44拍
        time_signatures.append(TimeSignature(0, 4, 4))
    elif time_signatures[0].time != 0:
        time_signatures[0].time = 0

    ends = [*(s.time for s in time_signatures[1:]), n_frame]

    weight = np.zeros(n_frame, dtype=np.float32)
    downbeat = np.zeros(n_frame, dtype=bool)
    for time_signature, end in zip(time_signatures, ends):
        start = int(time_signature.time)
        beats = time_signature.beats
        relative_index = np.arange(end - start, dtype=np.uint32)
        if beats % 3 == 0:  # 3 拍子
            index = relative_index % 3 == 0
            weight[start: end][~index] = 0.35
            downbeat[start: end][index] = True
        elif beats & (beats - 1) == 0:  # 2^n，处理为4 拍子
            weight[start: end][relative_index % 2 == 0] = 0.2
            weight[start: end][relative_index % 4 == 2] = 0.15
            downbeat[start: end][relative_index % 4 == 0] = True
        else:
            raise AssertionError(f"time signature: {time_signature} is invalid!")

    return downbeat, weight


@njit(cache=True)
def score_dp(cum_chroma, cum_bass, downbeat, weight):
    """
    使用动态规划解析出最佳对和弦排列
    """
    n_frame = cum_chroma.shape[1]
    cum_scores = np.full(n_frame, -np.inf)
    start_pos = np.zeros(n_frame, dtype=np.int32)
    final_choices = np.zeros(n_frame, dtype=np.int32)
    for i in range(n_frame):
        for j in range(MAX_PREV):
            if i - j < 0:
                break
            logits = chord_score(cum_chroma[j, i], cum_bass[j, i])
            best_choice = logits.argmax()
            score = logits[best_choice]
            if score < 0.2:
                score = 0.2
                best_choice = -1
            score += j * 0.7 + weight[i - j]
            pre_score = 0 if i - j == 0 else cum_scores[i - j - 1]
            cur_score = pre_score + score
            # 如果累计得分更高
            if cum_scores[i] < cur_score:
                cum_scores[i] = cur_score
                final_choices[i] = best_choice
                start_pos[i] = i - j - 1
            if j > 0 and downbeat[i - j + 1]:  # downbeat
                break
    return final_choices, start_pos


def decode_chords(
        beat_chroma: np.ndarray, beat_bass: np.ndarray,
        time_signatures: List[TimeSignature]) -> pd.DataFrame:
    """
    对提取得到的 pitch 与 bass 的数值，进行打分，并使用动态规划解析出最佳对和弦排列

    **sliding window requires np.__version__ >= 1.20**

    :param beat_chroma: shape 为 [n_frame, 12]，每一拍的 pitch 特征，由 feature.extrac_chord_feature() 得到
    :param beat_bass: shape 为 [n_frame, 12]，每一拍的 bass 特征，由 feature.extrac_chord_feature() 得到
    :param time_signatures: 拍号序列，可以为空
    :return: 以 pd.DataFrame 的类型，返回解析出的和弦，时间单位为 1拍
    """
    n_frame = len(beat_bass)
    # 每一beat的累计得分
    beat_chroma_pad = np.pad(beat_chroma.astype(np.float32), ((MAX_PREV - 1, 0), (0, 0)))
    beat_bass_pad = np.pad(beat_bass.astype(np.float32), ((MAX_PREV - 1, 0), (0, 0)))

    cum_chroma = np.stack([
        np.sum(  # sliding window requires np.__version__ >= 1.20
            sliding_window_view(beat_chroma_pad[MAX_PREV - j:], j, axis=0),
            axis=-1
        ) for j in range(1, MAX_PREV + 1)
    ])

    cum_bass = np.stack([
        np.sum(sliding_window_view(beat_bass_pad[MAX_PREV - j:], j, axis=0), axis=-1)
        for j in range(1, MAX_PREV + 1)
    ])

    downbeat, weight = downbeat_and_score_weight(n_frame, time_signatures)
    final_choices, start_pos = score_dp(cum_chroma, cum_bass, downbeat, weight)
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
            pitch = chord_pitches[choice] if choice != -1 else []
            result.append([start, end, name, pitch])
        end = start - 1

    return pd.DataFrame(result[::-1], columns=['start', 'end', 'name', 'pitch'])
