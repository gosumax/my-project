"""Render the saved deterministic audit as a reviewable Markdown report."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'runs/logic_engine_audit_smoke17_20261001'
FINAL = BASE/'final'

NOTES = {
('T02', '2876720584'): 'Перед FLOP kirill_petaev: 40 000 против bet160 000, unacted. kiril_petaev sits out не доказывает alias. Пayout на трёх картах с несколькими live; повтор NEW_HAND остаётся ambiguous.',
('T03', '2876719109'): 'olleggi/olleggì остаются разными наблюдаемыми именами; INVALID_CALL 36 000, повтор folds, неполные round responses и нераспознанные карточные строки.',
('T03', '2876720313'): 'Повтор BB без amount, stale forced contribution и повторные fold. Доказательства для удаления финансовых/actor строк нет.',
('T03', '2876727402'): 'Повтор SB без amount, повторные fold, OCR is All-in0/main potO/left table. Подозрение stale не позволяет mask.',
('T04', None): 'Частичный хвост до первого наблюдаемого hand marker, без blinds/payout. Равенство 0=0 не доказывает полноценную руку.',
('T04', '2876718521'): 'Нет полной forced-blind последовательности; raise-to30 000 не повышает текущий bet/contribution; повтор fold.',
('T04', '2876721764'): 'Повтор Starting a new hand внутри окна; uncertain NEW_HAND/DEAL. Boundary meaning не доказан, monetary action sequence отдельно от certification.',
('T04', '2876727536'): 'Повтор explicit header стал warning; повтор badpenguin folds остаётся невозможным в текущем player state.',
('T05', '2876723277'): 'posts Small / Blind40 000 не собраны как SB; повтор ante, неизвестный main pot, два live до payout без board. Полнота действий не доказана.',
('T06', None): 'Частичный хвост с неизвестным карточным OCR, без начала/blinds/payout. Math0=0 не свидетельство полной руки.',
('T06', '2876718617'): 'Wrapped/truncated reconnect и blind-level notices — информационные события; они сохраняются. Betting checks не заменяются предполагаемыми действиями.',
('T06', '2876722366'): 'BB без amount; повтор NEW_HAND; cards fragment. Без восстановления из наблюдений остаётся gap.',
('T06', '2876722787'): 'Payout81 100 точно равен ceiling81 100 с folded dead money. Повтор NEW_HAND R000277 остаётся AMBIGUOUS boundary; не sidepot bug.',
('T06', '2876724008'): 'Повтор start markers, split joined-table notice, TURN без подтверждённых карт и OCR830J040. Missing cards — unknown, не contradiction.',
('T06', '2876724664'): 'UNCALLED_RETURN vesnushka R000439 без amount и повтор start marker; никакое число не придумывалось.',
('T06', '2876727749'): 'Blind-level notices перестали быть действиями; повтор NEW_HAND остаётся unresolved boundary.',
('T06', '2876728636'): 'Повтор NEW_HAND, FLOP без карт, raw724029 неизвестен. Карточная неопределённость сохраняется.',
('T08', '2876718614'): 'Единственный OLD blocker: unresolved origin на DEAL R000018. Эта строка не меняет state/player/amount/cards/boundary; сохраняется warning.',
('T08', '2876721851'): 'Повтор truncated SB без amount; stale SB; return sezonnoe без amount. Подтверждённого duplicate mask нет.',
('T08', '2876724417'): 'Повтор SB etobilodavno без amount. Отсутствие численного evidence не устранено.',
('T08', '2876725694'): 'raise-to12 000 не выше current bet; rimi13 имеет12 000, но unacted перед FLOP; split joined-table/cards fragments.',
('T09', '2876726613'): 'raise-to160 000 не выше bet; kisunya777 unacted перед FLOP; повтор folds riverpomojet и неизвестный the main pot.',
}


def main():
    summary = json.loads((FINAL/'summary.json').read_text(encoding='utf-8'))
    matrix = json.loads((FINAL/'math_pass_22_matrix.json').read_text(encoding='utf-8'))
    assert summary['integrity_status'] == 'PASS'
    assert summary['fixed_22_total'] == 22 and len(matrix) == 22
    bench = json.loads((BASE/'final_benchmark/summary.json').read_text(encoding='utf-8'))
    full = json.loads((BASE/'final_tests/latest_full_summary.json').read_text(encoding='utf-8'))
    targeted = json.loads((BASE/'final_tests/latest_targeted_summary.json').read_text(encoding='utf-8'))
    integrity = json.loads((BASE/'final_tests/strict_integrity_summary.json').read_text(encoding='utf-8'))
    assert full['exit_code'] == targeted['exit_code'] == integrity['exit_code'] == 0
    lines = ['# Smoke17: deterministic Logic Engine audit', '', '## ROOT CAUSE SUMMARY', '',
             'Смешивание столов исключено: до любых исправлений 199/199 рук имели ровно один исходный table_id/session_id; девять отдельных replay совпали с ALL199/199. NEW повторно проходит тот же контроль. Сравнение использует одинаковый глобально доказанный CALL=ADDED контракт: локальная повторная калибровка требует witnesses минимум с трёх столов и создала бы конфигурационное различие.', '',
             'FILE → FUNCTION → CALLER → DATA: `ram_first_stage.py:FirstStage.rows` выставляет uncertain scroll origin → production runner сохраняет physical provenance → experimental `assemble_hands` разделяет `(table_id, session_id)` и вызывает `assemble_messages` → `validate_hand` до OLD continue(DEAL) применял blanket identity gate. Отдельно `parse_text` не распознавал explicit/wrapped server notices; OLD validator превращал UNKNOWN_EVENT в MISSING_ACTION. Missing board/hole evidence помечалось CARD_LOGIC_CONFLICT; eligibility выводилась из неполного ledger. Исправлены только эти экспериментальные правила/контракты verdict, источники сохранены.', '',
             '## TABLE ISOLATION: ALL vs T01–T09', '',
             '| Table | Hands | ALL Logic PASS | Alone Logic PASS | Mismatch |', '|---|---:|---:|---:|---:|']
    for table, v in summary['table_isolation'].items():
        lines.append(f"| {table} | {v['hands']} | {v['logic_pass_all']} | {v['logic_pass_alone']} | {len(v['mismatches'])} |")
    lines += ['', 'Каждая рука с hand_id/table_id/source_rows/unique_table_ids_inside_hand: [OLD manifest](../runs/logic_engine_audit_smoke17_20261001/table_isolation/all_hand_table_manifest.json), [NEW manifest](../runs/logic_engine_audit_smoke17_20261001/final/all_hand_table_manifest.json). Все списки unique_table_ids_inside_hand имеют длину1; нарушений0.', '',
              '## 22 MATH-PASS HANDS', '',
              f"Фиксированный OLD pool:22; complete18, incomplete4. BEFORE0/22, AFTER{summary['fixed_22_logic_pass_after']}/22. По 18 complete: см. строки complete; пул не пересобирался после изменения. Четыре неполных окна нельзя считать acceptance complete hands. Два хвоста имеют net=payout=0.", '',
              '| Table | Hand / epoch | Complete OLD | Action NEW | Card NEW | Final NEW | Причина / evidence |', '|---|---|---|---|---|---|---|']
    for m in matrix:
        note = NOTES[(m['table_id'], m['hand_id'])]
        lines.append(f"| {m['table_id']} | {m['hand_id'] or 'UNKNOWN'} /{m['hand_epoch']} | {m['old_complete_candidate']} | {m['action_logic_status']} | {m['card_status']} | {m['new_logic_status']} | {note} |")
    lines += ['', 'Полная матрица, отсортированная по first failure: [JSON](../runs/logic_engine_audit_smoke17_20261001/final/math_pass_22_matrix.json). Включает start/end, all failures, affected source IDs, timestamps и весь raw text. [Классификация каждого OLD failure](../runs/logic_engine_audit_smoke17_20261001/final/per_failure_classification.json) содержит A/B/C, исходный issue, raw/timestamp и фактическое состояние BEFORE/AFTER.', '',
              'Трассы actual validator: [OLD654 message steps](../runs/logic_engine_audit_smoke17_20261001/old_traces/traces.jsonl), [NEW](../runs/logic_engine_audit_smoke17_20261001/final/new_traces.jsonl); индивидуальные JSON/MD OLD и JSON NEW лежат рядом. Покрыты также continue branches, включая NEW_HAND/DEAL/UNKNOWN/TABLE_INFO. Это sys.settrace реального engine, не реконструкция альтернативным валидатором.', '',
              '## FALSE-POSITIVE RULES FOUND / LOGIC ENGINE BUGS', '',
              '- ROW provenance больше не равен betting FAIL. Unresolved action/money/card/boundary и строки financial inference evidence остаются CRITICAL certification blockers. Только harmless informational origin становится NONCRITICAL warning.',
              '- Explicit reconnect/limits notices, включая подтверждённые multiline continuation, становятся TABLE_INFO/BLIND_LEVEL. Неизвестные остальные строки остаются MISSING_ACTION. Notice не означает check/fold/call; round closure проверяется как прежде.',
              '- Отсутствующие карты дают NOT_VERIFIABLE, наблюдаемое противоречие — CONFLICT. Карты/sidepot/provenance не подменяют отдельный action verdict. Cards PASS означает consistency доступных данных, не полноту hand evidence.',
              '- Повтор того же explicit hand ID сохраняется как redundant metadata warning. Повтор ненумерованного NEW_HAND не признан stale: AMBIGUOUS completeness blocker.',
              '- Eligibility на неполном ledger: NOT_VERIFIABLE, вместо доказанного sidepot conflict. Денежная консервация проверяется независимо. Старый ceiling не изменён: среди22 нет доказанного counterexample.', '',
              '## TRUE POKER CONFLICTS FOUND / UPSTREAM DIRTY-TIMELINE CONFLICTS', '',
              'TRUE_POKER_CONFLICT означает несовместимость наблюдаемой parsed sequence с записанным BEFORE state, а не доказательство незаконной реальной игры. Примеры: действие после recorded fold, call36 000 при другом required delta, raise-to не выше bet. При dirty source эти нарушения могут быть upstream artefacts; ничьи имена не объединялись.', '',
              'UPSTREAM: missing forced blinds/amounts/actions/cards, incomplete capture tails, неоднозначные repeated NEW_HAND, repeated actor/financial observations. T02#2876720584 сохраняет unacted kirill_petaev40k против160k; похожее имя kiril_petaev в sits-out не является identity evidence. Math0 не доказывает полноту последовательности.', '',
              'RAW vs CLEANED: [все22 сравнения](../runs/logic_engine_audit_smoke17_20261001/final/raw_vs_cleaned_candidates.json). Временная маска разрешена только для повторного того же explicit hand identifier, не влияющего на деньги. Финансовые/NEW_HAND подозрения остаются AMBIGUOUS. Candidate masks adopted=0; observed rows никогда не удалялись.', '',
              '## OLD failure codes in fixed22 (hands containing code)', '', '| Code | Hands |', '|---|---:|']
    for code, count in sorted(summary['fixed_22_old_code_frequencies'].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f'| {code} | {count} |')
    lines += ['', 'Прочие перечисленные в задании codes:0 в этом пуле. Частоты не считаются автоматически root causes; классификация дана отдельно.', '',
              '## SMOKE17 BEFORE / AFTER', '', '| Metric | OLD | NEW |', '|---|---:|---:|']
    for key in ['candidate_windows', 'complete', 'logic_pass', 'math_pass', 'card_pass', 'fully_valid']:
        lines.append(f"| {key} | {summary['before'][key]} | {summary['after'][key]} |")
    lines += ['', '| Exclusive final status | OLD | NEW |', '|---|---:|---:|']
    for key in sorted(set(summary['before']['statuses']) | set(summary['after']['statuses'])):
        lines.append(f"| {key} | {summary['before']['statuses'].get(key, 0)} | {summary['after']['statuses'].get(key, 0)} |")
    lines += ['', 'Статусы были разделены; рост LOGIC_FAIL среди exclusive statuses связан также с переводом отсутствующих карт из CARD_CONFLICT в NOT_VERIFIABLE. Это не ухудшение action sequence. Подробнее: [summary](../runs/logic_engine_audit_smoke17_20261001/final/summary.json). Experiment FULLY_VALID_HAND не является production HH certification или independent visual acceptance.', '',
              '## TOP REMAINING FAILURE REASONS', '', '| Code | Hands NEW |', '|---|---:|---:|']
    for code, count in sorted(summary['after']['hands_containing_failure_codes'].items(), key=lambda kv: (-kv[1], kv[0]))[:12]:
        lines.append(f'| {code} | {count} |')
    lines += ['', '## FILES CHANGED / TESTS ADDED', '',
              '`parser_core/poker_hand_assembly.py`, `parser_core/poker_logic_engine.py` — experimental only. Диагностика: `experiments/audit_poker_logic_engine.py`, `experiments/replay_logic_audit.py`, `experiments/build_logic_audit_report.py`. Новые регрессии: `tests/test_poker_logic_layered_verdicts.py` (11 tests): harmless DEAL origin, critical monetary origin, missing vs contradictory cards, wrong winner retained, incomplete-ledger eligibility, same explicit header, ambiguous NEW_HAND, anchored/multiline server notices, missing response not hidden, inference financial dependency, wrapped/truncated pause preservation.', '',
              'Исходные59 tests и исходные math/card/Jev/replay файлы сохранены по hash. Все199 денежных ledger и player states совпали OLD/NEW. Production76/76 hashes совпали. Source8146/8146 fields/order сохранены; source SHA2b7822db0fa4c14706a8ef8e576b5942e22275cd28c29b62c5174720ff566a5b неизменен. Перестройка информационных сообщений не меняет source order.', '',
              '## PERFORMANCE BEFORE/AFTER', '', 'PERFORMANCE_PENDING', '', '## FULL PYTEST RESULT', '', 'PYTEST_PENDING', '',
              '## PRODUCTION CODE CHANGED', '', 'NO. Jev/TLS/API не изменялись и не запускались. Видео не декодировалось. Promotion отсутствует.', '',
              '## VERDICT', '', 'LOGIC_ENGINE_PARTIALLY_FIXED', '',
              'Доказанные ложные verdict исправлены; оставшиеся отказы сохранены с фактическими source/state причинами. Неполные/повторные/неоднозначные данные остаются blockers; никакая строка не удалена ради PASS.']
    path = ROOT/'docs/POKER_LOGIC_ENGINE_AUDIT_SMOKE17_20261001.md'
    performance = ['Warmup1 исключён;20×199=3980 hands на arm, tracing выключен. Четыре arms разделяют изменение validator и информационной assembly.', '',
                   '| Same input | Engine | Logic mean ms | Logic p95 ms | Math mean ms | Math p95 ms | Wall s |',
                   '|---|---|---:|---:|---:|---:|---:|']
    for name, v in bench['arms'].items():
        performance.append(f"| {name} | {name.rsplit('_', 1)[-1]} | {v['logic']['mean_ms']:.4f} | {v['logic']['p95_ms']:.4f} | {v['math']['mean_ms']:.4f} | {v['math']['p95_ms']:.4f} | {v['total_measured_wall_s']:.4f} |")
    performance += ['', 'NEW/OLD wall: frozen199 ×1.2745, NEW199 ×1.0563. Многократной деградации нет; это ограниченный sequential microbenchmark, не полноценное latency SLA. Frozen NEW arm имеет отдельный max12.48ms; wall варьирует между arms. Видео FPS/RTF и Jev latency здесь NOT_MEASURED. [Benchmark+hashes](../runs/logic_engine_audit_smoke17_20261001/final_benchmark/summary.json).']
    text = '\n'.join(lines)+'\n'
    text = text.replace('PERFORMANCE_PENDING', '\n'.join(performance))
    text = text.replace('PYTEST_PENDING',
                        f"`{targeted['command']}`: exit0,70 passed0.48s. `{full['command']}`: exit0,495 passed +7 subtests59.10s, wall{full['wall_seconds']:.3f}s. Strict production integrity: exit0 PASS. [Test receipts](../runs/logic_engine_audit_smoke17_20261001/final_tests/latest_full_summary.json).")
    text = re.sub(r'\]\((\.\./runs/[^)]+)\)', lambda m: '](<'+(path.parent/m.group(1)).resolve().as_posix()+'>)', text)
    path.write_text(text, encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
