from dataclasses import dataclass, field
from typing import Dict, List, Union, BinaryIO
from collections import defaultdict, OrderedDict
from .noteSet import Note, NoteSet, NOTE_ATTR_TYPE
from . import MSF_pb2 as msf
from . import midiToolkit

# 读取枚举类型对应的数值
_BEATS = msf.Sequence.GlobalChange.Type.BEATS
_BEATS_TYPE = msf.Sequence.GlobalChange.Type.BEATS_TYPE
_PITCH_BEND = msf.Track.TrackChange.Type.PITCH_BEND
_VOLUME = msf.Track.TrackChange.Type.VOLUME


@dataclass
class PitchBend:
    time: float
    pitch: int


@dataclass
class VolumeChange:
    time: float
    volume: int


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
    pitch_bends: List[PitchBend] = field(default_factory=list)
    volume_changes: List[VolumeChange] = field(default_factory=list)

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
class QPMChange:
    time: float
    qpm: int


@dataclass
class TimeSignature:
    time: float
    numerator: int = 4
    denominator: int = 4


@dataclass
class Sequence:
    """
    author: lyk
    status: available

    对应 trdparty.MSF_pb2 中的 Sequence, 可以实现与 msf、midi 文件的转换
    """
    meta: Dict[str, str] = field(default_factory=lambda: defaultdict(str))
    track: List[Track] = field(default_factory=list)
    qpm_changes: List[QPMChange] = field(default_factory=list)
    time_signature_changes: List[TimeSignature] = field(default_factory=list)

    def __post_init__(self):
        # 确保 attribute 为 DefaultDict
        assert isinstance(self.meta, dict), \
            f"The arg attribute should be Dict or DefaultDict, not {type(self.meta)}"
        if not isinstance(self.meta, defaultdict):
            temp = self.meta
            self.attribute = defaultdict(str)
            self.attribute.update(temp)

    @classmethod
    def from_msf(cls, file: Union[str, BinaryIO, msf.Sequence]):
        """
        类方法，读取 msf 文件，返回 Sequence 对象

        :param file: 文件路径，或者 BinaryIO
        :return: Sequence
        """
        if isinstance(file, str):
            with open(file, "rb") as f:
                sequence = msf.Sequence.FromString(f.read())
        elif isinstance(file, BinaryIO):
            sequence = msf.Sequence.FromString(file.read())
        elif isinstance(file, msf.Sequence):
            sequence = file
        else:
            raise AssertionError(f"type: {type(file)} is not supported when reading from MSF to Sequence")

        q = sequence.quantization

        # 从 GlobalChange 中读取拍号
        time_signatures = OrderedDict()
        beat_type = {_BEATS, _BEATS_TYPE}
        for change in filter(lambda x: x.type in beat_type, sorted(sequence.globalChange, key=lambda x: x.time)):
            time_signature = time_signatures.get(change.time, TimeSignature(change.time / q))
            if change.type == _BEATS:
                time_signature.numerator = int(change.value)
            else:  # _BEATS_TYPE
                time_signature.denominator = int(change.value)
            time_signatures[change.time] = time_signature

        return Sequence(
            meta={m.name: m.value for m in sequence.meta},
            qpm_changes=[
                QPMChange(qpm=change.value, time=change.time / q)
                for change in sequence.globalChange if change.type == msf.Sequence.GlobalChange.Type.QPM
            ],
            time_signature_changes=list(time_signatures.values()),
            track=[
                Track(
                    instrumentID=int(t.instrumentID),
                    meta={m.name: m.value for m in t.meta},
                    note=[
                        Note(
                            pitch=n.pitch,
                            start=n.start / q,
                            duration=n.duration / q,
                            attribute={NOTE_ATTR_TYPE[attr.type]: attr.value for attr in n.attribute}
                        )
                        for n in t.note
                    ],
                    pitch_bends=[
                        PitchBend(change.time / q, change.value)
                        for change in t.trackChange if change.type == _PITCH_BEND
                    ],
                    volume_changes=[
                        VolumeChange(change.time / q, change.value)
                        for change in t.trackChange if change.type == _VOLUME
                    ]
                )
                for t in sequence.track
            ]
        )

    def to_msf(self, quantization: int = 960) -> msf.Sequence:
        """
        将数据写入到 msf.Sequence 中，并返回该对象，方便进行下一步的传入或者写入操作。
        将 msf.Sequence 序列化并写入硬盘的方式如下：

        >>> with open("msf_filename.msf", 'wb') as f:
        >>>     f.write(sequence.SerializeToString())

        :param quantization: 将以 quarter 为单位的时间（float），乘以 quantization，并量化为 int 类型，便于存储
        :return: msf.Sequence
        """
        assert isinstance(quantization, int) and quantization > 0, f"quantization: {quantization} is invalid!"
        q = quantization
        # 整合 globalChange
        globalChange = [
            msf.Sequence.GlobalChange(
                type=msf.Sequence.GlobalChange.Type.QPM, time=int(qpm.time * q), value=qpm.qpm)
            for qpm in self.qpm_changes
        ]
        for time_signature in self.time_signature_changes:
            time = int(time_signature.time * q)
            globalChange.extend([
                msf.Sequence.GlobalChange(type=_BEATS, time=time, value=time_signature.numerator),
                msf.Sequence.GlobalChange(type=_BEATS_TYPE, time=time, value=time_signature.denominator)
            ])
        # 整合 trackChange
        trackChanges = []
        for track in self.track:
            trackChange = [
                msf.Track.TrackChange(type=_PITCH_BEND, time=int(change.time * q), value=change.pitch)
                for change in track.pitch_bends
            ]
            trackChange.extend([
                msf.Track.TrackChange(type=_VOLUME, time=int(change.time * q), value=change.volume)
                for change in track.volume_changes
            ])
            trackChanges.append(sorted(trackChange, key=lambda change: change.time))

        sequence = msf.Sequence(
            meta=[
                msf.Sequence.Meta(name=name, value=value)
                for name, value in self.meta.items()
            ],
            quantization=q,
            globalChange=sorted(globalChange, key=lambda x: x.time),
            track=[
                msf.Track(
                    instrumentID=int(track.instrumentID),
                    meta=[msf.Track.Meta(name=name, value=value) for name, value in track.meta.items()],
                    note=[
                        msf.Note(
                            pitch=n.pitch, start=int(n.start * q), duration=int(n.duration * q),
                            attribute=n.msf_attribute()
                        )
                        for n in track.note
                    ],
                    trackChange=trackChange
                )
                for track, trackChange in zip(self.track, trackChanges)
            ]
        )
        return sequence

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
            qpm_changes=[QPMChange(qpm=int(change.tempo), time=change.time / q) for change in midi.tempo_changes],
            time_signature_changes=[
                TimeSignature(change.time / q, change.numerator, change.denominator)
                for change in midi.time_signature_changes
            ],
            track=[
                Track(
                    instrumentID=int(instr.program),
                    meta={'name': instr.name, "is_drum": str(instr.is_drum)},
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
                    pitch_bends=[
                        PitchBend(change.time / q, change.pitch)
                        for change in instr.pitch_bends
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
            midiToolkit.TempoChange(tempo=change.qpm, time=int(change.time * q))
            for change in self.qpm_changes
        ]
        midi.time_signature_changes = [
            midiToolkit.TimeSignature(change.numerator, change.denominator, int(change.time * q))
            for change in self.time_signature_changes
        ]

        for track in self.track:
            instr = midiToolkit.Instrument(
                program=int(track.instrumentID) if track.instrumentID < 127 else 0,  # 将溢出值归 0
                name=track.meta['name'],
                is_drum=bool(track.meta['is_drum'])
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
                midiToolkit.PitchBend(pitch=change.pitch, time=int(change.time * q))
                for change in track.pitch_bends
            ]
            midi.instruments.append(instr)
        return midi
