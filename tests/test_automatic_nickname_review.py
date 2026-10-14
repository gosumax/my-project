import json
from types import SimpleNamespace
from scripts import review_nicknames as review

def cases(tmp_path):
    source=tmp_path/'pipeline';source.mkdir()
    (source/'nickname_review_cases.json').write_text(json.dumps(dict(cases=[dict(case_id='case')])),'utf8')
    return source

def test_no_pending_cases_does_not_launch(tmp_path,monkeypatch):
    source=cases(tmp_path)
    monkeypatch.setattr(review,'prepare_cases',lambda cases:(cases,{}))
    monkeypatch.setattr(review,'unreviewed_cases',lambda cases,registry:[])
    monkeypatch.setattr(review.subprocess,'Popen',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Unexpected server')))
    assert review.launch_cached_review(source) is None
    assert not (source/'nickname_review_launch.json').exists()

def test_pending_review_uses_verified_cached_route_and_separate_output(tmp_path,monkeypatch):
    source=cases(tmp_path);calls=[]
    monkeypatch.setattr(review,'prepare_cases',lambda cases:(cases,{}))
    monkeypatch.setattr(review,'unreviewed_cases',lambda cases,registry:cases)
    def popen(command,**kwargs):calls.append((command,kwargs));return SimpleNamespace(pid=123)
    monkeypatch.setattr(review.subprocess,'Popen',popen)
    result=review.launch_cached_review(source)
    command,options=calls[0]
    assert '--cases-file' in command and '--open-browser' in command
    assert command[command.index('--review-nicknames')+1]==str(source)
    assert command[command.index('--output')+1]!=str(source)
    assert result['pending']==1 and result['pid']==123
    assert (source/'nickname_review_launch.json').is_file()
