from datetime import datetime

from comparator import manifest


def _rec(model="hrrr", var="TMP", verif="rtma", mode="single",
         valid="2026-06-18T12:00:00", ts="2026-06-18T13:00:00", **kw):
    return manifest.make_record(
        ts_utc=ts, model=model, var=var, verif=verif, mode=mode,
        path=kw.get("path", f"/out/{model}_{var}.png"),
        cycle_dt=kw.get("cycle_dt"), fxx=kw.get("fxx"),
        valid_dt=valid, mean=kw.get("mean"), rmse=kw.get("rmse"), n=kw.get("n"),
    )


def test_make_record_iso_formats_datetimes():
    rec = manifest.make_record(
        ts_utc=datetime(2026, 6, 18, 13, 0, 5), model="hrrr", var="TMP",
        verif="rtma", mode="single", path="/out/x.png",
        cycle_dt=datetime(2026, 6, 17, 12, 0), fxx=24,
        valid_dt=datetime(2026, 6, 18, 12, 0), mean=1.0, rmse=2.0, n=10,
    )
    assert rec["ts_utc"] == "2026-06-18T13:00:05"
    assert rec["cycle_dt"] == "2026-06-17T12:00:00"
    assert rec["valid_dt"] == "2026-06-18T12:00:00"
    assert rec["fxx"] == 24


def test_append_and_read_roundtrip(tmp_path):
    mpath = tmp_path / "runs.jsonl"
    manifest.append(mpath, _rec())
    manifest.append(mpath, _rec(var="WIND"))
    rows = manifest.read(mpath)
    assert len(rows) == 2
    assert {r["var"] for r in rows} == {"TMP", "WIND"}


def test_append_creates_parent_dir(tmp_path):
    mpath = tmp_path / "nested" / "runs.jsonl"
    manifest.append(mpath, _rec())
    assert mpath.exists()


def test_read_filters(tmp_path):
    mpath = tmp_path / "runs.jsonl"
    manifest.append(mpath, _rec(model="hrrr", var="TMP"))
    manifest.append(mpath, _rec(model="nam12k", var="TMP"))
    manifest.append(mpath, _rec(model="hrrr", var="WIND", mode="gif"))
    assert {r["var"] for r in manifest.read(mpath, model="hrrr")} == {"TMP", "WIND"}
    assert len(manifest.read(mpath, model="nam12k")) == 1
    assert len(manifest.read(mpath, mode="gif")) == 1


def test_read_since_until_on_valid_dt(tmp_path):
    mpath = tmp_path / "runs.jsonl"
    manifest.append(mpath, _rec(valid="2026-06-17T00:00:00"))
    manifest.append(mpath, _rec(valid="2026-06-18T00:00:00"))
    manifest.append(mpath, _rec(valid="2026-06-19T00:00:00"))
    rows = manifest.read(mpath, since="2026-06-18", until="2026-06-18")
    assert len(rows) == 1
    assert rows[0]["valid_dt"].startswith("2026-06-18")


def test_read_sorted_newest_first_and_limit(tmp_path):
    mpath = tmp_path / "runs.jsonl"
    manifest.append(mpath, _rec(ts="2026-06-18T10:00:00"))
    manifest.append(mpath, _rec(ts="2026-06-18T12:00:00"))
    manifest.append(mpath, _rec(ts="2026-06-18T11:00:00"))
    rows = manifest.read(mpath, limit=2)
    assert [r["ts_utc"] for r in rows] == ["2026-06-18T12:00:00", "2026-06-18T11:00:00"]


def test_read_missing_file_returns_empty(tmp_path):
    assert manifest.read(tmp_path / "nope.jsonl") == []


def test_read_skips_torn_lines(tmp_path):
    mpath = tmp_path / "runs.jsonl"
    manifest.append(mpath, _rec())
    with open(mpath, "a") as fh:
        fh.write("{not valid json\n")
    assert len(manifest.read(mpath)) == 1
