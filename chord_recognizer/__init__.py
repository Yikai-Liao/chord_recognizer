import pandas as pd
from .feature import extract_chord_features
from .decode import decode_chords
from .parse.midiToolkit import MidiFile
from .parse import Sequence


def recognize_chords(path: str, note_precision: float = 0.25) -> pd.DataFrame:
    """
    给定 midi 文件的路径，返回识别的和弦的 DataFrame

    :param path: midi文件的路径
    :param note_precision: 在提取特征时，对 note 的时间相关参数进行量化的精度，单位为1拍，
        例如 note_precision=0.25, start = 1.5 会被量化为 6
    :return: 以 pd.DataFrame 的类型，返回解析出的和弦，时间单位为 1拍
    """
    midi = MidiFile(path)
    midi.instruments = list(filter(lambda instr: not instr.is_drum, midi.instruments))
    s = Sequence.from_midi(midi)
    return decode_chords(*extract_chord_features(s.track, note_precision))
