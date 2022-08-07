import numpy as np


def gen_chord_config():
    QUALITIES = {
        #       1     2     3     4  5     6     7
        'maj': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        'min': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        'aug': [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        'dim': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        'sus4': [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        'sus4(b7)': [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        'sus4(b7,9)': [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        'sus2': [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        '7': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        'maj7': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        'min7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        'minmaj7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        'maj6': [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        'min6': [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
        '9': [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        'maj9': [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        'min9': [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        '7(#9)': [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
        'maj6(9)': [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        'min6(9)': [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0],
        'maj(9)': [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        'min(9)': [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        'min(11)': [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
        '11': [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
        'maj9(11)': [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
        'min11': [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
        '13': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        'maj13': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        'min13': [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        'dim7': [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        'hdim7': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    }

    INVERSIONS = {
        'maj': ['3', '5'],
        'min': ['b3', '5'],
        '7': ['3', '5', 'b7'],
        'maj7': ['3', '5', '7'],
        'min7': ['5', 'b7'],
    }
    NUM_TO_ABS_SCALE = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    NUM_TO_INVERSION = ['1', 'b2', '2', 'b3', '3', '4', 'b5', '5', '#5', '6', 'b7', '7']
    INVERSION_TO_NUM = {inv: i for i, inv in enumerate(NUM_TO_INVERSION)}
    BASS_TEMPLATE = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    chord_name = []
    chromas = []
    basses = []
    inverse = []
    for i in range(12):
        for q_name, quality in QUALITIES.items():
            chord_name.append(f"{NUM_TO_ABS_SCALE[i]}:{q_name}")
            chroma = np.roll(quality, shift=i)
            chromas.append(chroma)
            basses.append(np.roll(BASS_TEMPLATE, shift=i))
            inverse.append(False)
            # 如果存在转位
            for inv in INVERSIONS.get(q_name, []):  # 让查询不到的情况直接跳过
                delta_scale = INVERSION_TO_NUM[inv]
                chord_name.append(f"{NUM_TO_ABS_SCALE[i]}:{q_name}/{inv}")
                chromas.append(chroma)
                basses.append(np.roll(BASS_TEMPLATE, shift=i + delta_scale))
                inverse.append(True)
    chord_name.append("N")
    chromas = np.array(chromas, dtype=bool)
    chromas_sum = chromas.sum(axis=1).astype(np.uint8)
    basses = np.array(basses, dtype=bool)
    chord_name = np.array(chord_name)
    inverse = np.array(inverse)
    return {'name': chord_name, 'bass': basses,
            'chroma': chromas, 'chroma_sum': chromas_sum,
            'score_bias': (-0.1 * chromas_sum - 0.05 * inverse).astype(np.float32)}
