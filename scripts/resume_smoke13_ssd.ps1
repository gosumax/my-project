$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\Новый парсер'
$python = 'C:\Users\Admin\AppData\Local\Programs\Python\Python313\python.exe'
$run = 'C:\ParserData\runs\smoke13_full_streaming_ssd_20260929_v1'
& $python -X utf8 'scripts\run_nine_tables_streaming.py' --video 'video\smoke_13.mp4' --output $run --max-frames 12148 --capture-workers 4 --tesseract-workers 5 --db-batch-frames 25 --resume *> (Join-Path $run 'runner_resume.log')
exit $LASTEXITCODE
