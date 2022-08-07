from dataclasses import dataclass, field
from typing import Dict, List
# from ..trdparty import MSF_pb2 as msf

# 顺序需要与 MSF 中保持一致，后续增加新的 attribute 需要在此处跟进
NOTE_ATTR_TYPE = ("ONSET_VEL", "OFFSET_VEL", "LEGATO", "PORTAMENTO")
NOTE_ATTR_NAME2NUM = {name: i for i, name in enumerate(NOTE_ATTR_TYPE)}


@dataclass
class Note:
    """
    author: lyk
    status: available

    对应 trdparty.MSF_pb2 中的 Note。

    * 为了减少转换开销，将原本动态表示的 attribute 全部提取为 Note 的属性，并且不在 __repr__中显示。
    * 重载了 __add__ 函数，使其支持使用 + 操作符进行移调。
    * 实现了 offset 函数，可以对 note 的 start 进行偏移
    """

    pitch: int
    start: float  # 时间的单位为一拍，无需考虑tempo
    duration: float
    attribute: Dict = field(default_factory=dict)

    def offset(self, time: int):
        self.start += time
        return self

    def __add__(self, num: int):
        self.pitch += num
        return self

    @property
    def velocity(self) -> int:
        """
        :return: 返回 note 的 onset_vel，如无效，默认值为 100
        """
        return self.attribute.get("ONSET_VEL", 100)

    # def msf_attribute(self) -> List[msf.Note.Attribute]:
    #     return [
    #         msf.Note.Attribute(type=NOTE_ATTR_NAME2NUM[name], value=value)
    #         for name, value in self.attribute.items()
    #     ]


@dataclass
class NoteSet:
    """
    author: lyk
    status: available

    对应 trdparty.MSF_pb2 中的 NoteSet。

    * 重载了 __add__ 函数，使其支持使用 + 操作符进行移调。
    * 实现了 offset 函数，可以对 note 的 start 进行偏移
    * 实现了 append 和 extend 函数，可以方便地对 note 列表进行拓展
    """
    note: List[Note] = field(default_factory=list)

    def append(self, note: Note):
        self.note.append(note)

    def extend(self, note_set):
        self.note.extend(note_set.note)

    def offset(self, time: int):
        for n in self.note:
            n.offset(time)
        return self

    def __add__(self, num: int):
        assert type(num) == int, "'+' operator for Note, NoteSet and Track only support int object"
        for n in self.note:
            n.pitch += num
        return self


