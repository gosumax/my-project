"""Report review availability honestly; never manufacture visual judgments."""
import json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/text_first_precision_review_smoke17_20261001'
def read(p):return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line]

def main():
    sample=json.loads((OUT/'sample_summary.json').read_text('utf-8'))
    sealed=read(OUT/'sealed_evidence.jsonl')
    root=read(OUT/'root_preliminary_visual.jsonl')
    ph=read(OUT/'post_hoc_results.jsonl')
    for r in ph:
        r.update(POST_HOC_RISK_RESULT='FRAGMENTATION_COMPATIBLE_NOT_PROVEN',independent_validation_complete=False,
            root_visual_observation=('Calls446000 target remains between bet446000 above and Dealing Turn/cards below in causal endpoints and returned-old frame; roster/folds persist above.' if r['case_id']=='V175' else
                'Shows target remains between Full house/cards above and Three of a kind/cards below; River and win lines append in later viewport.'),
            possible_alternate_explanation='One visible occurrence received a new track ID during viewport motion, then old ID was inherited again; later old ID does not alone prove a new occurrence.',
            false_same_established=False,causal_verdict_unchanged=True,
            limitation='Root preliminary observation, not independent review; intervening future frames not exhaustively inspected. Neither false SAME nor full identity truth is certified.')
    (OUT/'post_hoc_results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ph),encoding='utf-8')
    independent=[]
    for suffix in ('A','B'):
        p=OUT/f'visual_{suffix}.jsonl'
        if p.exists():independent.extend(read(p))
    # Failed calls yielded no saved final labels; render receipts exist. Keep a distinct pending state rather than
    # pretending incomplete reviews received VISUALLY_UNCERTAIN judgments.
    assert not independent,'Review results became available: aggregate them before reporting pending.'
    records=[dict(case_id=r['case_id'],pair_id=r['pair_id'],review_status='NOT_REVIEWED',visual_label=None,
        precision_included=False,blocking_condition='INDEPENDENT_REVIEWER_USAGE_LIMIT',original_simulation_verdict=r['original_simulation_verdict'],
        categories=r['categories']) for r in sealed]
    (OUT/'review_status.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    summary=dict(status='INDEPENDENT_VISUAL_VALIDATION_BLOCKED',sample_total=len(sealed),
        candidate_sample=280,ambiguous_sample=50,independent_reviewed_total=0,visually_same=0,visually_different=0,
        visually_uncertain=0,not_reviewed=len(sealed),confirmed_precision=None,precision_status='NOT_MEASURED_DENOMINATOR_ZERO',
        root_preliminary_total=len(root),root_preliminary_counts=dict(Counter(r['label'] for r in root)),root_labels_precision_eligible=False,
        category_precision={cat:dict(selected=sum(cat in r['categories'] and r['original_simulation_verdict']=='TEXT_FIRST_SAME_CANDIDATE' for r in sealed),reviewed=0,SAME=0,DIFFERENT=0,UNCERTAIN=0,precision=None)
            for cat in ('NORMAL','ANTE','SB_BB','FOLD','NICK_ACTION','GAP2','WEAK_CONTENT','MAX_Y_RESIDUAL','POST_HOC_RISK')},
        observed_false_negative_candidates=None,false_negative_sample_reviewed=0,false_positive_validation='NOT_MEASURED',
        post_hoc_results=ph,edge_before=6685,edge_after=6685,validated_relationships=0,edge_simulation_applied=False,
        image_writes=0,source_data_changed=False,production_code_changed=False,
        verdict='TEXT_FIRST_PRECISION_NOT_ENOUGH',
        blocked_reviewers=[dict(reviewer=x,error='You have hit your usage limit',final_labels_saved=0,rendering_may_have_completed=True) for x in ('visual_a','visual_b')],
        task_complete=False,remaining='Independent blinded RAM visual review of280candidates/50stratifiedAMBIGUOUS, categoryprecision and observedfalse-negative adjudication.')
    (OUT/'precision_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# TEXT-FIRST precision validation — Smoke17', '',
        '## FILES CHANGED','',
        'Новые экспериментальные scripts: `experiments/text_first_precision_review.py`, `experiments/report_text_first_precision_review.py`; JSON/JSONL review artifacts под `runs/text_first_precision_review_smoke17_20261001/`; этот отчёт. Ни одного production/source/classifier файла не изменено.', '',
        '## REVIEW SAMPLE','',
        '**330 случаев подготовлено: все280 prospective SAME candidates и50 stratified AMBIGUOUS.** Каждый pair имеет уникальный case_id, категории перекрываются без дублирования cases. Admission label/metrics спрятаны от reviewer; order фиксирован SHA256(pair_id), watermark classification=VISUAL_REVIEW_PENDING.', '',
        '| Candidate category | selected | independently reviewed |','|---|---:|---:|']
    for cat,v in summary['category_precision'].items():lines.append(f'| {cat} | {v["selected"]} | 0 |')
    lines+=['',
        'NORMAL154 означает полное покрытие обычных candidates. WEAK_CONTENT28 — top decile по близости target к observed local controls envelope; остальные candidates тоже включены. MAX_Y_RESIDUAL106 — всеcandidate residual1.5px. GAP2 candidates0: все7 gap2 присутствуют вAMIBGUOUS sample. ANTE58/SB_BB13/FOLD15/NICK_ACTION55 включены полностью.', '',
        'AMBIGUOUS sample включает15 near-admission,15 duplicated text,10 no-mapping,15 weak-content,15 normal ивсе7 gap2; membership overlap, unique sample50. Стратификация фиксирована до review. Это prepared coverage, не completed review coverage.', '',
        'В `sealed_evidence.jsonl` для каждого case: stable IDs/table/session, causalendpointframes/exactPTS, delta, normalizedtext/uniqueness, expectedactualy/residual/shift, same-IDneighborlist, targetMAE/XOR/controlenvelopes,registered rowdistance, contentalignment/bbox/trims, boundary/overlap/gap/bridge/categories. `blind_cases.jsonl` не содержит admissionlabel, scores или futurelifetimes. Разметка сделана только наRAM-копиях; native source arrays readonly и hashes сверены.', '',
        '## VISUAL VALIDATION RESULT','',
        '**Независимая validation не выполнена.** Оба separate reviewer завершились с сервисной ошибкой лимита использования без сохранённых итоговых labels. Receipts подтверждают часть RAM-показов до сбоя; они не доказывают завершённую классификацию. Не запущены OCR, model substitutions или автоматический adjudicator.', '',
        '| Independent review status | COUNT |','|---|---:|',
        '| reviewed total | 0 |','| VISUALLY_SAME | 0 |','| VISUALLY_DIFFERENT | 0 |','| VISUALLY_UNCERTAIN | 0 |','| NOT_REVIEWED / pending | 330 |', '',
        'NOT_REVIEWED здесь означает отсутствие завершённой независимой классификации; часть изображений была показана, но итоговых labels нет. Этот статус не подменяется VISUALLY_UNCERTAIN. Пустые independent counts не означают отсутствие ошибок. Четыре случая V001/V002/V135/V175 фактически показаны root вRAM и получили preliminary VISUALLY_SAME поtarget+neighborcontext; root знает архитектуру/результат admission и не считается независимым validator. Их4/4 нельзя выдавать за precision.', '',
        '## PRECISION','',
        '`SAME/(SAME+DIFFERENT)` = **NOT_MEASURED**, denominator0. UNCERTAIN исключается изформулы, pendingтоже. Overall/category precision NORMAL/ANTE/SB_BB/FOLD/NICK_ACTION/GAP2/WEAK_CONTENT/POST_HOC_RISK: **NOT_MEASURED**. Нетconfidenceinterval/100%precision/certifiedSAME.', '',
        '## FALSE POSITIVES','',
        '**NOT_MEASURED.** Ни одного independently adjudicated false-positive/true-positive. Сохраняются217previouslyblockedduplicate risks; эти riskflagsне groundtruthfalsemerges. SameOCR/targetscoresне используютсякакvalidationtruth.', '',
        '## AMBIGUOUS FALSE-NEGATIVE SAMPLE','',
        'Prepared50 / independentlyreviewed0. `observed false-negative candidates`: **NOT_MEASURED**. Нетоснованияобъявить0false negatives; false-negative/recallвывод остаётсяpending.', '',
        '## POST_HOC REAPPEARANCE','',
        'Causal root preliminary labels V135/V175 записаны до показа futureframes. Ни causal TEXT-FIRST verdict, ни thresholds не изменены.', '',
        '| CASE | old→new IDs | new first | old returns | delay |','|---|---|---:|---:|---:|']
    for r in ph:lines.append(f'| {r["case_id"]} | {r["old_track_id"].split(":")[-1]} → {r["new_track_id"].split(":")[-1]} | {r["new_first_frame"]} ({r["new_first_ms"]:.3f}ms) | {r["old_reappearance_frame"]} ({r["old_reappearance_ms"]:.3f}ms) | {r["delta_after_new_frames"]}frames / {r["delta_after_new_ms"]:.3f}ms |')
    lines+=['',
        'V135: shows line иThree-of-a-kind continuation/cards сохраняются между предыдущимFull-house/cardblock иследующимriverblock. V175: calls446000 находится послеbet446000 идоDealingTurn/cardblock; тажесоседняяструктура видна при возвратеoldID. На просмотренных endpoints/reappearanceframes нет положительного visualдоказательствановогоoccurrence. Это совместимо сfragmentedtrackeridentity.', '',
        '**POST_HOC_RISK_RESULT для обоих: FRAGMENTATION_COMPATIBLE_NOT_PROVEN.** FalseSAMEнеустановлен, но полныйfutureinterval и независимая adjudicationне завершены. OldIDreappearanceсампо себе незадаётGT; оба случая остаются withheld отofflineacceptance. Future evidenceнеподаётсявcausalverdict.', '',
        '## EDGE-GATE PROSPECTIVE IMPACT','',
        '**6685 → 6685 UNRESOLVED_EDGE**, independentlyvalidatedrelationships0. Никакогофактическогоreclassification. Existinggateне потребляет experimentalrelationships; hypotheticalcountremovalбезphysicalvalidationиnewgatecontractнепридумывается. Rows8146,IDs/sourcefieldsнеизменны.', '',
        '## IMAGE WRITES','', '**0.** SourceROI/overlay/zoom/sheetencodeтолькоRAM(BytesIO); нетPNG/JPG/WEBPфайловилиdiskbase64imagecache. JSONrenderreceiptsсодержатhash/frame/provenanceбезimagepayload. Before/afterinventoryиinputhashauditзаписанв`safety_audit.json`.', '',
        '## SOURCE DATA CHANGED','', 'NO.', '', '## PRODUCTION CODE CHANGED','', 'NO.', '',
        '## VERDICT','', '**TEXT_FIRST_PRECISION_NOT_ENOUGH** — independentprecisionнеизмерена. Это невыводTEXTFIRSTUNSAFE и непроверка280«безошибочных»links.', '',
        'Validation остаётсянезавершённойиз-залимитаseparatereviewer. Дляпродолженияготовы330blindcases,assignmentA/B(20sharedcases),sealedcausalevidenceиRAMrenderer. Ничегоневнедрено, ниодинrule/thresholdнеизменён.']
    (ROOT/'docs/TEXT_FIRST_PRECISION_REVIEW_SMOKE17_20261001.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('sample_total','independent_reviewed_total','not_reviewed','confirmed_precision','root_preliminary_total','verdict','task_complete')}))

if __name__=='__main__':main()
