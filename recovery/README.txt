Полное восстановление production на Windows x64

Требуются Windows x64, PowerShell 5.1+, Git и установленный шрифт Arial.
Установленные Python, Node и OCR не требуются. Используйте новый короткий
путь назначения, например C:\PDrestore. Не используйте рабочую production-папку.

Полный клон (после полного скачивания восстановление работает без сети):
git clone https://github.com/gosumax/my-project.git C:\PDclone
powershell -NoProfile -ExecutionPolicy Bypass -File C:\PDclone\recovery\restore.ps1 -Destination C:\PDrestore

Для нестабильной сети можно получать проверяемые части по отдельности:
git -c http.version=HTTP/1.1 clone --depth 1 --filter=blob:none --sparse https://github.com/gosumax/my-project.git C:\PDclone
git -C C:\PDclone sparse-checkout set --no-cone /recovery/restore.ps1 /recovery/restore.py /recovery/run.ps1 /recovery/run.py /recovery/recovery_manifest.json /recovery/README.txt
powershell -NoProfile -ExecutionPolicy Bypass -File C:\PDclone\recovery\restore.ps1 -Destination C:\PDrestore

Частичный клон требует сети до получения всех пакетов и исходников.
Git-запросы частей ограничены по времени; SHA256 и размер проверяются.
Повторная команда использует только помеченное как принадлежащее восстановлению
назначение; неизвестная непустая папка отклоняется. Не изменяйте клон между
восстановлением и проверкой. Для приватного репозитория нужен доступ GitHub.

Проверка восстановленного комплекта:
powershell -NoProfile -ExecutionPolicy Bypass -File C:\PDclone\recovery\restore.ps1 -Destination C:\PDrestore -VerifyOnly

Ограниченный официальный запуск со своим исходным видео:
powershell -NoProfile -ExecutionPolicy Bypass -File C:\PDclone\recovery\run.ps1 -Destination C:\PDrestore -Video C:\Video\input.mp4 -Frames 20 -Output C:\PDtest01

Путь Output должен быть новым. Видео не включено в резервную копию.
База ников не включена по прямому требованию владельца и не публикуется.
Исходная локальная база остаётся нетронутой. Восстановленная копия начинает
работу с пустым реестром ников.
Восстановление разворачивает отдельные среды main/OCR, Node, OCR-веса и кеш.
Нужная Paddle библиотека Microsoft OpenMP vcomp140.dll включена рядом с его
собственными DLL; установленный в Windows Visual C++ Runtime для неё не нужен.
Рабочий production не изменяется. Исходники алгоритмов сохраняются побайтно;
в восстановленной копии мигрируются только пути конфигурации и манифеста.
Оригиналы остаются в Git. Успешный запуск не является новым Golden,
сертификацией качества HH или измерением полного видео/RTF.
