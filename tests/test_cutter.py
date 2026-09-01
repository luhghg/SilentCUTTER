import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from silence_cutter.cutter import CutterError, build_filter_complex_script
from silence_cutter.detector import Segment


def test_build_filter_complex_script_single_segment_with_audio():
    script = build_filter_complex_script([Segment(1.0, 3.5)], has_audio=True)
    assert "[0:v]trim=start=1.000000:end=3.500000,setpts=PTS-STARTPTS[v0]" in script
    assert "[0:a]atrim=start=1.000000:end=3.500000,asetpts=PTS-STARTPTS[a0]" in script
    assert "[v0][a0]concat=n=1:v=1:a=1[outv][outa]" in script


def test_build_filter_complex_script_no_audio():
    script = build_filter_complex_script([Segment(0.0, 2.0)], has_audio=False)
    assert "[0:a]" not in script
    assert "[v0]concat=n=1:v=1:a=0[outv]" in script
    assert "[outa]" not in script


def test_build_filter_complex_script_multiple_segments_order_and_labels():
    segments = [Segment(0.0, 1.0), Segment(2.0, 3.0), Segment(5.0, 6.0)]
    script = build_filter_complex_script(segments, has_audio=True)
    for i in range(3):
        assert f"[v{i}]" in script
        assert f"[a{i}]" in script
    assert "[v0][a0][v1][a1][v2][a2]concat=n=3:v=1:a=1[outv][outa]" in script


def test_build_filter_complex_script_empty_segments_raises():
    with pytest.raises(CutterError):
        build_filter_complex_script([], has_audio=True)
