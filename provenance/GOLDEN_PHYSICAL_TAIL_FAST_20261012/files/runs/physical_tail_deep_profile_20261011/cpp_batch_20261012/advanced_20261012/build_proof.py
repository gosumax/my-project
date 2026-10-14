"""Build full-proof candidate in its own DLL; identical AVX2 FP flags."""
from pathlib import Path
import subprocess
P=Path(__file__).resolve().parent
(P/'wholeproof.cpp').write_text((P/'native_threads.cpp').read_text()+'\n#include "proof_core.hpp"\n')
flags=['c++','-std=c++17','-O2','-fno-fast-math','-fno-associative-math','-ffp-contract=off','-fno-vectorize','-fno-slp-vectorize','-fno-unroll-loops','-march=x86_64','-mavx2','-mno-fma','-shared']
r=subprocess.run([str(P.parent/'toolchain/zig-windows-x86_64-0.13.0/zig.exe'),*flags,str(P/'wholeproof.cpp'),'-o',str(P/'wholeproof.dll')],capture_output=True,text=True)
(P/'proof_build.log').write_text(r.stdout+r.stderr)
print('fullproof compile exit',r.returncode,flush=True)
if r.returncode:print(r.stderr[-5000:])
raise SystemExit(r.returncode)
