import numpy as np
from .config import CHORD_CONFIG
from typing import Union, List, Tuple

array_like = Union[np.ndarray, List, Tuple]


def chord_score(chroma: array_like, bass: array_like) -> np.ndarray:
    """
    调用 chord_score_batch 实现对单个frame对特征计算

   :param chroma: shape 为 [12]， 数值在 0～1 之间，对应和弦的 pitches
   :param bass: shape 为 [12]，数值在 0~1 之间，但是每一个特征向量只有一个非0值，对应核心的 bass
   :return: 返回该在每一个种类上的得分，shape为 [num_classes]，区间为[-inf, inf]
   """
    chroma = np.asarray(chroma, dtype=np.float32)
    bass = np.asarray(bass, dtype=np.float32)
    return chord_score_batch(chroma.reshape(1, 12), bass.reshape(1, 12))[0]


def chord_score_batch(chroma: array_like, bass: array_like) -> np.ndarray:
    """
    根据 chroma 与 bass 特征，批量计算每一个frame 对每一类和弦的得分。

    * 该函数是提取和弦时的主要耗时部分
    * 该函数能够识别的和弦种类，取决于配置文件中标注的和弦 **转位** 以及 **性质** 数量
    * 配置文件由 config.gen_config.gen_chord_config() 函数生成，并提前保存为 npz 文件存储在本地以加速程序

    :param chroma: shape 为 [batch, 12]， 数值在 0～1 之间，对应和弦的 pitches
    :param bass: shape 为 [batch, 12]，数值在 0~1 之间，但是每一个特征向量只有一个非0值，对应核心的 bass
    :return: 返回每个frame在每一个种类上的得分，shape为 [batch, num_classes]，区间为[-inf, inf]
    """
    ref_chroma: np.array = CHORD_CONFIG['chroma']
    ref_chroma_sum: np.array = CHORD_CONFIG['chroma_sum']
    ref_bass: np.array = CHORD_CONFIG['bass']
    score_bias: np.array = CHORD_CONFIG['score_bias']
    chroma = np.asarray(chroma, dtype=np.float32).reshape((-1, 1, 12))
    bass = np.asarray(bass, dtype=np.float32).reshape((-1, 1, 12))

    score_chroma = (np.sum(chroma * ref_chroma, axis=2) - np.sum(chroma * ~ref_chroma, axis=2)) / ref_chroma_sum
    score_bass = np.sum((0.5 * bass) * ref_bass, axis=2)
    score = score_chroma + (score_bass + score_bias)
    return score
