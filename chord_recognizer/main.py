import pandas as pd
from .feature import extract_chord_features
from .decode import decode_chords
from .parse.midiToolkit import MidiFile
from .parse import Sequence
from typing import Union
from os.path import splitext, isfile


def recognize_chords(file: Union[str, MidiFile, Sequence], note_precision: float = 0.25) -> pd.DataFrame:
    """
    给定 midi 文件的路径，返回识别的和弦的 DataFrame

    :param file: 文件路径，或者已经实例化的 MidiFile, Sequence 类
    :param note_precision: 在提取特征时，对 note 的时间相关参数进行量化的精度，单位为1拍，
        例如 note_precision=0.25, start = 1.5 会被量化为 6
    :return: 以 pd.DataFrame 的类型，返回解析出的和弦，时间单位为 1拍
    """
    if isinstance(file, str):
        assert isfile(file), f"{file} is not a file!"
        ext = splitext(file)[-1][1:]
        if ext == "msf":
            s = Sequence.from_msf(file)
        elif ext in {'mid', "MID"}:
            s = Sequence.from_midi(file)
        else:
            raise AssertionError(f"Do not support {ext} file!")
    elif isinstance(file, MidiFile):
        s = Sequence.from_midi(file)
    elif isinstance(file, Sequence):
        s = file
    else:
        raise AssertionError(f"the file argument do not support type: {type(file)}!")

    return decode_chords(*extract_chord_features(s.track, note_precision))
