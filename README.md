# Chord Recognizer

一个快速的 midi 和弦提取工具

* 目前对于测试 midi，可以在达到 100ms 量级 的识别速度
* 使用了 numba 进行 jit 加速，因此第一次运行时可能速度较慢
* 仍处于开发阶段，后续仍会提速，并提升识别精度和鲁棒性
* 已经将识别功能封装为一个函数，可以在 `example.ipynb` 中看到使用方法
