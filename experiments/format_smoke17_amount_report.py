"""Presentation only: index unabridged reports and separate amount scopes/controls."""
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/smoke17_amount_materiality_20261001'


def main():
    text=(OUT/'FULL_TRANSCRIPTS.md').read_text('utf-8')
    chunks=re.split(r'(?m)(?=^## (?:RAW|CORE)_\d{3}:)',text)[1:]
    hands=OUT/'hands';hands.mkdir(exist_ok=False)
    for chunk in chunks:
        view=re.match(r'## ((?:RAW|CORE)_\d{3}):',chunk).group(1)
        (hands/(view+'.md')).write_text(chunk,encoding='utf-8')
    rows=json.loads((OUT/'table.json').read_text('utf-8'))
    amount=set();controls=set()
    for row in rows:
        view=re.search(r'\[((?:RAW|CORE)_\d{3})\]',row['HAND']).group(1)
        (controls if row['PLAYER_ACTION'].startswith('NO CALL') else amount).add(view)
    groups=json.loads((OUT/'groups.json').read_text('utf-8'))
    focused={kind:{cat:[] for cat in 'ABC'} for kind in groups}
    control_groups={kind:[] for kind in groups}
    index=['# Полный индекс OCR/event transcripts','',
           'Every linked file contains its full physical OCR and DERIVED events. RAW and CORE are overlapping views, never distinct added hands.','']
    for kind,categories in groups.items():
        index+=['## '+kind,'']
        for cat,entries in categories.items():
            for item in entries:
                view=item['view_id']
                label='amount '+cat if view in amount else 'source/math control; NO_AMOUNT_KIND_DISPUTE'
                line=f'- [{view}: {item["table"]} / {item["hand"] or "UNKNOWN"}](hands/{view}.md) — {label}'
                index.append(line)
                (focused[kind][cat] if view in amount else control_groups[kind]).append(item)
        index.append('')
    (OUT/'TRANSCRIPT_INDEX.md').write_text('\n'.join(index)+'\n',encoding='utf-8')
    (OUT/'amount_groups.json').write_text(json.dumps(focused,ensure_ascii=False,indent=2),encoding='utf-8')
    for cat in 'ABC':
        lines=[f'# {cat}: amount-kind scopes only','']
        for kind,categories in focused.items():
            lines+=['## '+kind,'']
            for item in categories[cat]:
                view=item['view_id'];lines.append(f'- [{view}: {item["table"]} / {item["hand"] or "UNKNOWN"}](hands/{view}.md) — {item["reason"]}')
            lines.append('')
        (OUT/(cat+'_AMOUNT_HANDS.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines=['# Source/math controls without an ADDED/TOTAL action-kind candidate','',
           'These scopes are included for full Smoke17 coverage and sizing/math/identity inspection. Their presence does not assert an ADDED/TOTAL dispute. See full transcripts for actual missing facts.','']
    for kind,entries in control_groups.items():
        lines+=['## '+kind,'']
        for item in entries:
            view=item['view_id'];lines.append(f'- [{view}: {item["table"]} / {item["hand"] or "UNKNOWN"}](hands/{view}.md)')
        lines.append('')
    (OUT/'SOURCE_CONTROLS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    summary=json.loads((OUT/'summary.json').read_text('utf-8'))
    summary['amount_scope_group_counts']={kind:{cat:len(v) for cat,v in categories.items()} for kind,categories in focused.items()}
    summary['no_amount_kind_control_counts']={kind:len(v) for kind,v in control_groups.items()}
    summary['individual_full_transcripts']=len(chunks)
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    report=(OUT/'REPORT.md').read_text('utf-8')
    report+='\n## Amount scopes vs source/math controls\n\n'+json.dumps(summary['amount_scope_group_counts'],ensure_ascii=False)+'\n\n'
    report+='73 RAW and17 CORE scopes have no action-kind candidate. They are separate source/math controls, not new ADDED/TOTAL disputes. A_AMOUNT_HANDS.md, B_AMOUNT_HANDS.md, C_AMOUNT_HANDS.md list amount scopes only; SOURCE_CONTROLS.md lists the rest. TRANSCRIPT_INDEX.md links every unabridged hand view.\n'
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps(summary['amount_scope_group_counts']))


if __name__=='__main__':main()
