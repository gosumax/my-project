"""Bounded state-rule replay of unchanged, source-bound per-frame NCC receipts."""
import argparse,collections,hashlib,itertools,json,re,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from dealing_persistent_hand_20261003 import PersistentHandStateTrigger
V3=ROOT/'runs/start_stack_two_stage_ncc_20261003/hand_state_v3'
HEADER=re.compile(r'^(?:Hand\s*)?#\s*(\d{8,})\s*$')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def read(path):
    with path.open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():yield json.loads(line)
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def run(mode,out):
    start=time.perf_counter();out.mkdir(parents=True,exist_ok=True)
    old=mode=='old20'
    source=V3/({'old20':'regression20_fenced','smoke17_first':'smoke17_replay','smoke17_heldout':'heldout_new'}[mode])
    filename='rows.jsonl' if old else 'row_evaluations.jsonl';first=1801 if mode=='smoke17_first' else 3601
    header_path=source/('header_pixel_checks.jsonl' if old else 'source_pixel_controls.jsonl')
    manifest={'scope':'Requested reused-sample persistent-hand state regression, not fresh heldout',
              'mode':mode,'source_rows':str(source/filename),'source_rows_sha256':digest(source/filename),
              'header_controls_sha256':digest(header_path),'core_sha256':digest(ROOT/'experiments/dealing_persistent_hand_20261003.py'),
              'runner_sha256':digest(Path(__file__)),'ncc_config_sha256':digest(V3.parent/'candidate_v2/frozen_config.json'),
              'classifier_reexecuted':False,'classifier_receipts_unchanged':True,'production_changed':False,
              'live_context_scope':'Fixed recorded forward source/session and table layout; no confirmed replacement/history interaction receipts in these fixtures. This does not certify a production live-context detector.'}
    dump(out/'manifest.json',manifest)
    headers=collections.defaultdict(list)
    for r in read(header_path):
        if old:
            if r['pixel_verified']:headers[(r['frame'],r['table'])].append((r['hand_id'],r['row_index'],None))
        elif r['pass']:
            match=HEADER.fullmatch(r['raw_ocr_text'].strip())
            if match:headers[(r['frame'],r['table'])].append((match.group(1),None,r['actual_sha256']))
    state=PersistentHandStateTrigger();seen={};triggers=[];counts=collections.Counter();lastsample=None;transitions=[];t09=[]
    key=lambda r:(r['table'],r.get('sample_hand_id'),r['frame'])
    with (out/'frame_receipts.jsonl').open('w',encoding='utf-8') as dest:
        for (table,sample,frame),bucket in itertools.groupby(read(source/filename),key=key):
            rows=list(bucket)
            if old and (table,sample)!=lastsample:
                transitions.extend(state.transitions);state=PersistentHandStateTrigger();seen={};lastsample=(table,sample)
            session='smoke14_frozen_sample' if old else 'smoke17_fixed_source'
            state.set_context(table,session,str(table),frame,confirmed=True)
            confirmed=[]
            for hand,index,pixels in headers.get((frame,table),[]):
                row=next((r for r in rows if (r['row_index']==index if old else r['pixel_hash']==pixels)),None)
                if row is None:continue
                digest_value=row['pixel_hash'];seen[(table,digest_value)]=hand
                if (old and hand==sample) or (not old and frame>first):confirmed.append((row['row_bounds'][0],hand,digest_value))
            # Multiple independently visible headers can be legitimate history
            # above the newest one. Latest chronological segment is bottommost.
            if confirmed:
                _,hand,pixels=max(confirmed,key=lambda x:x[0]);state.confirm_hand(table,hand,frame,pixels)
            for row in rows:row['hand_header_id']=seen.get((table,row['pixel_hash'])) or row.get('hand_header_id')
            before=state.current_hand_id(table)
            event,reason,eligible=state.observe(table,frame,rows,live_context_confirmed=True)
            record={'table':table,'frame':frame,'sample_hand_id':sample,'current_hand_before':before,
                    'current_hand_after':state.current_hand_id(table),'latch_reason':reason,'eligible_rows':list(eligible)}
            dest.write(json.dumps(record,ensure_ascii=False)+'\n');counts[reason]+=1
            counts['classification_UNKNOWN']+=sum(r['status']=='UNKNOWN' for r in rows)
            counts['stage1_excluded']+=sum(not r['gate_pass'] for r in rows)
            counts['stage2_rows']+=sum(r['scores'] is not None for r in rows)
            if event:
                event['source_pts']=rows[0].get('source_pts');triggers.append(event)
            if table==9 and 4358<=frame<=4362:
                record=dict(record,cards=[{k:r.get(k) for k in ('pixel_hash','row_index','status','gate_part_scores','scores','margin')} for r in rows if r.get('status')=='cards'])
                t09.append(record)
    transitions.extend(state.transitions)
    for name,data in [('triggers',triggers),('state_transitions',transitions)]:
        with (out/(name+'.jsonl')).open('w',encoding='utf-8') as f:
            for r in data:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    if t09:dump(out/'t09_trace.json',t09)
    result={'mode':mode,'trigger_count':len(triggers),'counts':dict(counts),'wall_seconds':time.perf_counter()-start,
            'DUPLICATE_TRIGGERS':len(triggers)-len({(r['table'],r['session_id'],r['table_instance_id'],r['context_generation'],r['hand_id']) for r in triggers}),
            'UNKNOWN_table_frames':counts['CURRENT_HAND_UNKNOWN'],'production_changed':False,'metrics_status':'AWAITING_INDEPENDENT_EVENT_REVIEW'}
    dump(out/'summary.json',result);return result

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['old20','smoke17_first','smoke17_heldout'],required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(run(args.mode,args.output),ensure_ascii=False,indent=2))
