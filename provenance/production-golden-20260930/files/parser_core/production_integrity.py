"""Manifest enforcement for the registered RAM-first pipeline; no recognition logic."""
from __future__ import annotations
import ast
import hashlib
import importlib.abc
import importlib.machinery
import json
import subprocess
import sys
import sysconfig
from pathlib import Path

class ProductionIntegrityError(RuntimeError):
    pass

def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def load_manifest(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'production_manifest.json').read_text('utf8'))
    required={'production_version','git_commit','entrypoint','internal_runner','modules','configs',
              'exporter','tests','hashes','png_policy','performance','stages','forbidden_import_roots'}
    if not required<=manifest.keys():
        raise ProductionIntegrityError('Invalid production manifest: missing '+str(sorted(required-manifest.keys())))
    for p in manifest['modules']+manifest['configs']+[manifest['entrypoint'],manifest['exporter']]:
        candidate=(root/p).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ProductionIntegrityError('Missing/outside production file: '+p)
    if manifest['entrypoint']!='run_production.py' or manifest['internal_runner']!='scripts/run_smoke14_ram.py':
        raise ProductionIntegrityError('Registered production entrypoint substitution')
    if manifest['pipeline']!='RAM_FIRST' or manifest['card_path']!='PIXEL_ONLY':
        raise ProductionIntegrityError('Production pipeline policy substitution')
    if not (root/'PRODUCTION.md').is_file():
        raise ProductionIntegrityError('Missing PRODUCTION.md')
    if not (root/'PRODUCTION.md').read_text('utf8').startswith('THIS FILE DEFINES THE ONLY PRODUCTION PIPELINE.'):
        raise ProductionIntegrityError('Invalid production instruction header')
    return manifest

def import_targets(tree,source_path=None):
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            for name in node.names:yield name.name
        elif isinstance(node,ast.ImportFrom):
            base=node.module or ''
            if node.level and source_path:
                package=Path(source_path).parent.parts
                prefix='.'.join(package[:len(package)-node.level+1])
                base='.'.join(x for x in (prefix,base) if x)
            if base:
                yield base
                for alias in node.names:
                    if alias.name!='*':yield base+'.'+alias.name

def registered_files(manifest):
    return set(manifest['modules'])|set(manifest['configs'])|set(manifest.get('allowed_dependencies',[]))|{manifest['entrypoint'],manifest['exporter']}

def validate_integrity(root,manifest=None,*,strict_hashes=False):
    root=Path(root).resolve();manifest=manifest or load_manifest(root)
    allowed=registered_files(manifest);errors=[];modified=[]
    for path,digest in manifest['hashes'].items():
        full=(root/path).resolve()
        if not full.is_relative_to(root) or not full.is_file():errors.append('Missing registered file: '+path)
        elif sha256(full)!=digest:modified.append(path)
    exceptions=set(manifest.get('allowed_dependencies',[]))
    for path in sorted(allowed):
        full=root/path
        if not full.is_file():errors.append('Missing production dependency: '+path);continue
        if full.suffix!='.py':continue
        tree=ast.parse(full.read_text('utf-8-sig'),filename=path)
        for target in import_targets(tree,path):
            candidate=target.replace('.','/')+'.py'
            package=target.replace('.','/')+'/__init__.py'
            local=next((p for p in (candidate,package) if (root/p).is_file()),None)
            forbidden=target.split('.')[0] in manifest['forbidden_import_roots']
            # From-import attributes are not separate Python modules; the
            # defining registered helper is checked independently.
            allowed_attribute = local is None and any(target.startswith(p[:-3].replace('/','.')+'.') for p in exceptions if p.endswith('.py'))
            if forbidden and (local is None or local not in exceptions) and not allowed_attribute:
                errors.append(f'Forbidden production import: {path} -> {target}')
            elif local and local not in allowed:
                errors.append(f'Unregistered production import: {path} -> {local}')
    for stage,entry in manifest['stages'].items():
        path=entry['file'];parts=entry['symbol'].split('.');symbol=parts[0]
        if path not in allowed:errors.append('Unregistered production stage: '+stage)
        elif path.endswith('.py'):
            tree=ast.parse((root/path).read_text('utf-8-sig'))
            if symbol not in {n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))}:
                errors.append(f'Missing stage symbol: {stage} -> {path}:{symbol}')
            elif len(parts)>1:
                owner=next((n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==symbol),None)
                if owner is None or parts[1] not in {n.name for n in owner.body if isinstance(n,ast.FunctionDef)}:
                    errors.append(f'Missing stage method: {stage} -> {path}:{entry["symbol"]}')
    if strict_hashes and modified:errors.append('Production manifest hashes stale: '+', '.join(modified))
    if errors:raise ProductionIntegrityError('\n'.join(errors))
    return modified

class RegisteredImportGuard(importlib.abc.MetaPathFinder):
    def __init__(self,root,manifest):
        self.root=Path(root).resolve();self.allowed=registered_files(manifest)
        self.forbidden=set(manifest['forbidden_import_roots'])
    def find_spec(self,fullname,path=None,target=None):
        first=fullname.split('.')[0]
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec is None:return None
        if spec.origin and spec.origin not in ('built-in','frozen'):
            resolved=Path(spec.origin).resolve()
            if not resolved.is_relative_to(self.root) and first in self.forbidden|{'parser_core','scripts'}:
                raise ProductionIntegrityError('Foreign production module: '+fullname+' -> '+str(resolved))
            if resolved.is_relative_to(self.root):
                relative=resolved.relative_to(self.root).as_posix()
                if relative not in self.allowed:
                    raise ProductionIntegrityError('Unregistered/legacy production module: '+relative)
        return None

class ProductionGuard:
    def __init__(self,root,manifest=None):
        self.root=Path(root).resolve();self.manifest=manifest or load_manifest(root)
        self.modified=validate_integrity(self.root,self.manifest)
        self.output=None;self.early_checked=False;self.import_guard=None;self.last_png_check_frame=-1
        self.metrics={}
    def fingerprint(self):
        m=self.manifest
        try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.root,text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.CalledProcessError):commit=None
        lines=['PARSER PD PRODUCTION','VERSION: '+m['production_version'],
               'COMMIT: '+str(commit),'GOLDEN COMMIT: '+str(m['git_commit']),
               'PIPELINE: '+m['pipeline'],'ENTRYPOINT: '+m['entrypoint'],
               'CARD PATH: '+m['card_path'],'NCC: '+m.get('ncc_mode','BACKUP'),
               'PNG TRANSPORT: DISABLED','EXPECTED RTF: '+str(m['performance']['expected_rtf_range']),
               'MANIFEST SHA256: '+sha256(self.root/'production_manifest.json')]
        if self.modified or commit!=m['git_commit']:
            lines+=['WORKTREE MODIFIED']+[('  '+p) for p in self.modified]
            if commit!=m['git_commit']:lines+=['  commit differs from golden']
        print('\n'.join(lines),flush=True)
        return dict(version=m['production_version'],commit=commit,golden_commit=m['git_commit'],
                    manifest_sha256=sha256(self.root/'production_manifest.json'),
                    modified_files=self.modified,pipeline=m['pipeline'],card_path=m['card_path'])
    def install_import_guard(self):
        self.import_guard=RegisteredImportGuard(self.root,self.manifest)
        # Also validate modules already imported before installing the finder.
        for name,module in list(sys.modules.items()):
            origin=getattr(module,'__file__',None)
            if not origin:continue
            p=Path(origin).resolve()
            if ((p.is_relative_to(self.root) and p.relative_to(self.root).as_posix() not in self.import_guard.allowed)
                    or (not p.is_relative_to(self.root) and name.split('.')[0] in {'parser_core','scripts','experiments'})):
                raise ProductionIntegrityError('Already-loaded unregistered production module: '+name)
        sys.meta_path.insert(0,self.import_guard)
        library_roots={Path(p).resolve() for key,p in sysconfig.get_paths().items()
                       if key in ('stdlib','platstdlib','purelib','platlib')}
        def check_exec(event,args):
            if event!='exec' or not args:return
            filename=getattr(args[0],'co_filename','')
            if not filename or filename.startswith('<'):return
            path=Path(filename).resolve()
            if path.is_relative_to(self.root):
                if path.relative_to(self.root).as_posix() not in self.import_guard.allowed:
                    raise ProductionIntegrityError('Unregistered production source execution: '+str(path))
            elif not any(path.is_relative_to(library) for library in library_roots):
                raise ProductionIntegrityError('Foreign production source execution: '+str(path))
        # Catches direct file loaders as well as ordinary imports. Registered
        # internal runner and installed Python libraries retain their real paths.
        sys.addaudithook(check_exec)
    def configure(self,args):
        bindings=self.manifest['config_arguments']
        for argument,path in bindings.items():
            if Path(getattr(args,argument)).resolve()!=(self.root/path).resolve():
                raise ProductionIntegrityError('Production config substitution: '+argument)
        self.output=Path(args.output).resolve()
        if self.output.exists():raise ProductionIntegrityError('Production output must be fresh')
    def check_png(self):
        if self.output is None or not self.output.exists():return dict(count=0,bytes=0)
        allowed=set(self.manifest['png_policy'].get('allowed_debug_paths',[]));files=[]
        for p in self.output.rglob('*'):
            if p.is_file() and p.suffix.lower()=='.png' and p.relative_to(self.output).as_posix() not in allowed:files.append(p)
        measured=dict(count=len(files),bytes=sum(p.stat().st_size for p in files))
        self.metrics['intermediate_png']=measured
        policy=self.manifest['png_policy']
        if measured['count']>policy['max_intermediate_count'] or measured['bytes']>policy['max_intermediate_bytes']:
            raise ProductionIntegrityError('MASS PNG WRITE DETECTED — POSSIBLE WRONG/LEGACY PIPELINE '+str(measured))
        return measured
    def check_early(self,frames,capture_wall,*,force=False):
        if frames-self.last_png_check_frame>=50 or force:
            self.check_png();self.last_png_check_frame=frames
        p=self.manifest['performance']
        if self.early_checked or frames<p['early_frames']:return
        self.early_checked=True
        expected=p['golden_capture_seconds']*frames/p['golden_frames']
        ratio=capture_wall/expected
        self.metrics['early_performance']=dict(frames=frames,capture_wall_seconds=capture_wall,
            fps=frames/capture_wall if capture_wall else None,golden_scaled_wall_seconds=expected,
            capture_rtf=capture_wall/(frames/p.get('golden_source_fps',30)),
            output_bytes=sum(f.stat().st_size for f in self.output.rglob('*') if f.is_file()) if self.output is not None and self.output.exists() else 0,
            regression_ratio=ratio,regression_factor=p['regression_factor'])
        print('PRODUCTION EARLY GATE: '+json.dumps(self.metrics['early_performance']),flush=True)
        if ratio>p['regression_factor']:
            message='PRODUCTION PERFORMANCE REGRESSION '+json.dumps(self.metrics['early_performance'])
            if self.modified:raise ProductionIntegrityError(message+'; modified production path')
            print(message+'; no legacy-path evidence; continuing with warning',flush=True)
    def finish(self,frames,capture_wall,wall,bytes_written):
        try:
            self.check_early(frames,capture_wall,force=True)
        except ProductionIntegrityError as exc:
            self.metrics['failure']=str(exc)
            raise
        finally:
            self.metrics['final']=dict(frames=frames,wall_seconds=wall,bytes_written=bytes_written,
                                       pipeline=self.manifest['pipeline'],card_path=self.manifest['card_path'])
            if self.output is not None:
                (self.output/'production_guard.json').write_text(json.dumps(self.metrics,indent=2),'utf8')
