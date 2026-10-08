from utils.fused_trace import prepare_fused_trace


def test_prepare_fused_trace_strips_bom_and_resolves_from_fused_dir(tmp_path):
    fused_dir = tmp_path / "fusedBin"
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    source = trace_dir / "input.csv"
    source.write_bytes(b"\xef\xbb\xbf0.162052\n\n0.160924\n")

    normalized = prepare_fused_trace("../traces/input.csv", fused_dir=fused_dir)

    assert normalized == fused_dir / "trace_input.csv"
    assert normalized.read_bytes() == b"0.162052\n0.160924\n"

