from typing import List, Optional, Tuple
import numpy as np

from .util import Note, Track

note_type = np.dtype([
    ('pitch', np.uint8),
    ('start', np.uint32),
    ('end', np.uint32)
])


def to_note_arr(notes: List[Note], precision: float) -> np.ndarray:
    return np.fromiter(
        filter(
            lambda x: x[-1] > 0,
            map(
                lambda x: (x.pitch, (x.start / precision) + 0.5, ((x.duration + x.start) / precision) + 0.5),
                notes
            )),
        dtype=note_type
    )


def get_abs_pianoroll(note_arr: np.ndarray, end: Optional[int] = None):
    time_len = note_arr['end'][-1] if not end else end
    count = np.zeros((12, time_len), dtype=np.uint8)
    pitch_names = note_arr['pitch'] % 12
    for pitch_name, start, end in zip(pitch_names, note_arr['start'], note_arr['end']):
        count[pitch_name, start: end] = 1
    return count.T


def get_bass(note_arr: np.ndarray, end: Optional[int] = None):
    time_len = note_arr['end'][-1] if not end else end
    bass = np.ones(time_len, dtype=np.uint8) * 128
    note_arr = np.sort(note_arr, order="pitch")
    for pitch, start, end in note_arr[::-1]:
        bass[start: end] = pitch
    return bass


def get_track_weight(abs_pianoroll: List, bass: List):
    thickness_mean = []
    bass_mean = []

    for p in abs_pianoroll:
        thickness = p.sum(axis=1)
        thickness = thickness[thickness.nonzero()]
        thickness_mean.append(thickness.mean() if len(thickness) > 0 else 0)
    for b in bass:
        total_len = len(b)
        b = b[b < 128]
        nonempty_rate = len(b) / total_len
        bass_mean.append(b.mean() if nonempty_rate > 0.2 else 128)

    weight = 1 - np.exp(0.95 - np.array(thickness_mean))
    weight /= weight.max()
    weight[np.argmin(bass_mean)] = 1
    return weight


def extract_chord_features(tracks: List[Track], note_precision: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """
    综合所有Track的信息，提取统一的和弦数值特征

    :param tracks: parse.sequence 中 Track 类的列表，不能是打击乐器
    :param note_precision: 在提取特征时，对 note 的时间相关参数进行量化的精度，单位为1拍，
        例如 note_precision=0.25, start = 1.5 会被量化为 6
    :return: 返回和弦的pitch特征，以及bass特征，shape均为 [batch, 12]，数值均在 0~1 之间
    """
    assert int(1 / note_precision) == 1 / note_precision
    chord_window = int(1 / note_precision)

    tracks = [
        to_note_arr(
            sorted(track.note, key=lambda note: -note.pitch),
            note_precision
        ) for track in tracks if track.meta.get('is_drum', False) == 'False'
    ]  # 数组按照 note.pitch 排序降序排列
    ends = (track['end'].max() for track in tracks)
    global_end = max(ends)
    global_end += chord_window - global_end % chord_window

    # 统计每个 track 的稠密度和 低音
    abs_pianoroll = [get_abs_pianoroll(track, global_end) for track in tracks]
    bass = [get_bass(track, global_end) for track in tracks]
    weight = get_track_weight(abs_pianoroll, bass)

    # 按照窗口切分，并求和，加权取极值
    window_shape = (len(tracks), -1, chord_window, 12)
    # [track, time, window, pitch_name] -> [track, time, pitch_name]
    chroma = np.stack(abs_pianoroll).reshape(window_shape).sum(axis=-2) * (weight / chord_window).reshape(-1, 1, 1)
    # [track, time, pitch_name] -> [time, pitch_name]
    chroma = chroma.max(axis=0)

    bass = np.min(bass, axis=0)
    bass_mask = bass == 128
    bass_chroma = np.eye(12)[bass % 12]
    bass_chroma[bass_mask] = 0
    # [time, window, pitch_name] -> [time, pitch_name]
    bass_chroma = bass_chroma.reshape(-1, chord_window, 12).sum(axis=-2) / chord_window
    return chroma, bass_chroma
