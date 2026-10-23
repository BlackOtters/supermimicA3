from io import StringIO

from a3_supermimic.scripts.recover_tracking import metric_rows


def test_incomplete_live_jsonl_row_is_retried_when_completed():
    stream = StringIO('{"iteration": 50, "wm/loss": 1.0}\n{"iteration": 100')
    assert list(metric_rows(stream)) == [(50, {"wm/loss": 1.0})]
    position = stream.tell()
    stream.seek(0, 2)
    stream.write(', "wm/loss": 0.5}\n')
    stream.seek(position)
    assert list(metric_rows(stream)) == [(100, {"wm/loss": 0.5})]
