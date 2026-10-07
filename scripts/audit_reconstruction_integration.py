"""Offline frozen-output parity audit; never a production launcher."""
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.poker_pipeline import analyze_timeline
from parser_core.poker_reconstruction_layer import write_reconstruction


def without_timings(value):
    if isinstance(value, dict):
        return {key: without_timings(item) for key, item in value.items()
                if key not in {'wall_seconds', 'elapsed_ms', 'logic_ms'}}
    if isinstance(value, list):
        return [without_timings(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    names = ['ordered_timeline.jsonl', 'raw_ocr_results.jsonl', 'physical_rows.jsonl']
    def hashes():
        return {n: hashlib.sha256((source / n).read_bytes()).hexdigest() for n in names}
    before_hashes = hashes()
    rows = [json.loads(line) for line in (source / names[0]).read_text('utf-8').splitlines()]
    statuses = {}
    for line in (source / names[1]).read_text('utf-8').splitlines():
        row = json.loads(line)
        statuses[row['row_id']] = row['raw_ocr_response'].get('status', 'ERROR')
    spec = importlib.util.spec_from_file_location('before_poker_pipeline',
        ROOT / 'runs/reconstruction_integration_20261003/before/parser_core/poker_pipeline.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    before, before_summary = old.analyze_timeline(rows, ocr_status=statuses)
    after, after_summary = analyze_timeline(rows, ocr_status=statuses)
    frozen = copy.deepcopy(after)
    summary = write_reconstruction(output, after)
    for name, data in [('before_hand_analysis.json', before), ('after_hand_analysis.json', after)]:
        (output / name).write_text(json.dumps(data, ensure_ascii=False), 'utf-8')
    before_summary.pop('wall_seconds'); after_summary.pop('wall_seconds')
    receipt = {'source': str(source.resolve()), 'timeline_rows': len(rows),
        'hands_seen': len(after), 'legacy_records_equal_except_timing':
            without_timings(before) == without_timings(after),
        'legacy_records_unmodified_by_layer': after == frozen,
        'excluded_timing_keys': ['wall_seconds', 'elapsed_ms', 'logic_ms'],
        'legacy_summary_equal_except_timing': before_summary == after_summary,
        'validated_hand_ids_equal': [r['segment_id'] for r in before if r['fully_validated']] ==
                                    [r['segment_id'] for r in after if r['fully_validated']],
        'input_hashes_before': before_hashes, 'input_hashes_after': hashes(),
        'reconstruction': summary, 'mode': 'FROZEN_FULL_TIMELINE_REPLAY; NO VIDEO/OCR/TRACKER RERUN'}
    (output / 'parity_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), 'utf-8')
    assert receipt['legacy_records_equal_except_timing'] and receipt['legacy_records_unmodified_by_layer']
    assert receipt['legacy_summary_equal_except_timing']
    assert before_hashes == receipt['input_hashes_after']
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
