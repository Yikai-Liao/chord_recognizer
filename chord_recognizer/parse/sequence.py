from dataclasses import dataclass, field
from typing import Dict, List, Union, BinaryIO
from collections import defaultdict
from .note_set import Note, NoteSet, NOTE_ATTR_TYPE
# from ..trdparty import MSF_pb2 as msf
from . import midiToolkit


# _BPM: int = msf.Sequence.GlobalChange.Type.BPM
# _PITCH_BEND: int = msf.Track.TrackChange.Type.PITCH_BEND
# _ONSET_VEL: int = msf.Note.Attribute.Type.ONSET_VEL


@dataclass
class TrackChange:
    type: str  # 使用枚举类型 TrackChangeType 来进行判断赋值
    time: float
    value: int


@dataclass
class Track:
    """
    author: lyk
    status: available

    对应 trdparty.MSF_pb2 中的 Track，

    * 重载了 + 运算符，便于进行移调
    * 实现了 append 和 extend 函数，便于增加 note
    """
    instrumentID: int = 1
    meta: Dict[str, str] = field(default_factory=lambda: defaultdict(str))
    note: List[Note] = field(default_factory=list)
    trackChange: List[TrackChange] = field(default_factory=list)

    def __post_init__(self):
        # 确保 attribute 为 DefaultDict
        assert isinstance(self.meta, dict), \
            f"The arg attribute should be Dict or DefaultDict, not {type(self.meta)}"
        if not isinstance(self.meta, defaultdict):
            temp = self.meta
            self.attribute = defaultdict(str)
            self.attribute.update(temp)

    def append(self, note: Note):
        self.note.append(note)

    def extend(self, note_set: NoteSet):
        self.note.extend(note_set.note)

    def __add__(self, num: int):
        """
        重载 + 运算符，对 Track 中的所有 Note 的 pitch 增加 num，并返回自身

        :param num: 要增加的 midi number
        :return: Track
        """
        assert type(num) == int, "'+' operator for Note, NoteSet and Track only support int object"
        for n in self.note:
            n.pitch += num
        return self


@dataclass
class GlobalChange:
    type: str  # 使用枚举类型 GlobalChangeType 来进行判断赋值
    time: float
    value: int


@dataclass
class Sequence:
    """
    author: lyk
    status: available

    对应 trdparty.MSF_pb2 中的 Sequence, 可以实现与 msf、midi 文件的转换
    """
    meta: Dict[str, str] = field(default_factory=lambda: defaultdict(str))
    globalChange: List[GlobalChange] = field(default_factory=list)
    track: List[Track] = field(default_factory=list)

    def __post_init__(self):
        # 确保 attribute 为 DefaultDict
        assert isinstance(self.meta, dict), \
            f"The arg attribute should be Dict or DefaultDict, not {type(self.meta)}"
        if not isinstance(self.meta, defaultdict):
            temp = self.meta
            self.attribute = defaultdict(str)
            self.attribute.update(temp)

    # @classmethod
    # def from_msf(cls, file: Union[str, BinaryIO, msf.Sequence]):
    #     """
    #     类方法，读取 msf 文件，返回 Sequence 对象
    #
    #     :param file: 文件路径，或者 BinaryIO
    #     :return: Sequence
    #     """
    #     if isinstance(file, str):
    #         with open(file, "rb") as f:
    #             sequence = msf.Sequence.FromString(f.read())
    #     elif isinstance(file, BinaryIO):
    #         sequence = msf.Sequence.FromString(file.read())
    #     elif isinstance(file, msf.Sequence):
    #         sequence = file
    #     else:
    #         raise AssertionError(f"type: {type(file)} is not supported when reading from MSF to Sequence")
    #
    #     q = sequence.quantization
    #     return Sequence(
    #         meta={m.name: m.value for m in sequence.meta},
    #         globalChange=[
    #             GlobalChange(type=change.type, time=change.time / q, value=change.value)
    #             for change in sequence.globalChange
    #         ],
    #         track=[
    #             Track(
    #                 instrumentID=int(t.instrumentID),
    #                 meta={m.name: m.value for m in t.meta},
    #                 note=[
    #                     Note(
    #                         pitch=n.pitch,
    #                         start=n.start / q,
    #                         duration=n.duration / q,
    #                         attribute={NOTE_ATTR_TYPE[attr.type]: attr.value for attr in n.attribute}
    #                     )
    #                     for n in t.note
    #                 ],
    #                 trackChange=[
    #                     TrackChange(type=change.type, time=change.time / q, value=change.value)
    #                     for change in t.trackChange
    #                 ]
    #             )
    #             for t in sequence.track
    #         ]
    #     )
    #
    # def to_msf(self, quantization: int = 960) -> msf.Sequence:
    #     """
    #     将数据写入到 msf.Sequence 中，并返回该对象，方便进行下一步的传入或者写入操作。
    #     将 msf.Sequence 序列化并写入硬盘的方式如下：
    #
    #     >>> with open("msf_filename.msf", 'wb') as f:
    #     >>>     f.write(sequence.SerializeToString())
    #
    #     :param quantization: 将以 quarter 为单位的时间（float），乘以 quantization，并量化为 int 类型，便于存储
    #     :return: msf.Sequence
    #     """
    #     assert isinstance(quantization, int) and quantization > 0, f"quantization: {quantization} is invalid!"
    #     q = quantization
    #     sequence = msf.Sequence(
    #         meta=[
    #             msf.Sequence.Meta(name=name, value=value)
    #             for name, value in self.meta.items()
    #         ],
    #         quantization=q,
    #         globalChange=[
    #             msf.Sequence.GlobalChange(type=change.type, time=int(change.time * q), value=change.value)
    #             for change in self.globalChange
    #         ],
    #         track=[
    #             msf.Track(
    #                 instrumentID=int(t.instrumentID),
    #                 meta=[msf.Track.Meta(name=name, value=value) for name, value in t.meta.items()],
    #                 note=[
    #                     msf.Note(
    #                         pitch=n.pitch, start=int(n.start * q), duration=int(n.duration * q),
    #                         attribute=n.msf_attribute()
    #                     )
    #                     for n in t.note
    #                 ],
    #                 trackChange=[
    #                     msf.Track.TrackChange(type=change.type, time=int(change.time * q), value=change.value)
    #                     for change in t.trackChange
    #                 ]
    #             )
    #             for t in self.track
    #         ]
    #     )
    #     return sequence

    @classmethod
    def from_midi(cls, file: Union[str, BinaryIO, midiToolkit.MidiFile]):
        """
        类方法，读取 midi 文件，返回 Sequence 对象

        :param file: 文件路径，或者 BinaryIO，或者 midiToolkit.MidiFile
        :return: Sequence
        """
        if isinstance(file, str):
            midi = midiToolkit.MidiFile(filename=file)
        elif isinstance(file, BinaryIO):
            midi = midiToolkit.MidiFile(file=file)
        elif isinstance(file, midiToolkit.MidiFile):
            midi = file
        else:
            raise AssertionError(f"type: {type(file)} is not supported when reading from MIDI to Sequence")

        q = midi.ticks_per_beat
        return Sequence(
            globalChange=[
                GlobalChange(type="BPM", time=change.time / q, value=int(change.tempo))
                for change in midi.tempo_changes
            ],
            track=[
                Track(
                    instrumentID=int(instr.program),
                    meta={'name': instr.name},
                    note=[
                        Note(
                            pitch=note.pitch,
                            start=note.start / q,
                            duration=(note.end - note.start) / q,
                            attribute={"ONSET_VEL": note.velocity}
                        )
                        for note in instr.notes
                    ],
                    # 忽略 midi 中的 control change 信息，因为没有被包含在 msf 中
                    trackChange=[  # 读取 midi 中的 pitch_bends
                        TrackChange(type="PITCH_BEND", value=int(bend.pitch), time=bend.time / q)
                        for bend in instr.pitch_bends
                    ]
                )
                for instr in midi.instruments
            ]
        )

    def to_midi(self, quantization: int = 960):
        """
        将数据写入到 midiToolkit.MidiFile 中，并返回该对象，方便进行下一步的传入或者写入操作。
        将 MidiFile 写入硬盘或者序列化的方式为

        >>> midi.dump(filename="midi_filename.mid")
        >>> with open("midi_filename.mid", "wb") as f:
        >>>     midi.dump(file=f)

        :param quantization: 将以 quarter 为单位的时间（float），乘以 quantization，并量化为 int 类型，便于存储
        :return: midiToolkit.MidiFile
        """
        assert isinstance(quantization, int) and quantization > 0, f"quantization: {quantization} is invalid!"
        q = quantization
        midi = midiToolkit.MidiFile(ticks_per_beat=quantization)
        midi.tempo_changes = [
            midiToolkit.TempoChange(tempo=change.value, time=int(change.time * q))  # 遍历globalChange中所有的 BPM Change
            for change in filter(lambda x: x.type == "BPM", self.globalChange)
        ]

        for track in self.track:
            instr = midiToolkit.Instrument(
                program=int(track.instrumentID) if track.instrumentID < 127 else 0,  # 将溢出值归 0
                name=track.meta['name']
            )
            instr.notes = [
                midiToolkit.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=int(note.start * q),
                    end=int((note.start + note.duration) * q),
                )
                for note in track.note
            ]
            instr.pitch_bends = [
                midiToolkit.PitchBend(pitch=change.value, time=int(change.time * q))
                for change in filter(lambda x: x.type == "PITCH_BEND", track.trackChange)
            ]
            midi.instruments.append(instr)
        return midi
