"""Render the verified current Smoke17 blocker inventory, without changing engines."""
from collections import Counter, defaultdict
from decimal import Decimal
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'runs/current_blockers_smoke17_20261001'
RULES = {
    'ROW_IDENTITY_UNRESOLVED': ('Происхождение строки', 'У критического сообщения есть source row с origin=UNRESOLVED_IDENTITY. Не доказано, новая это запись или повтор наблюдения при scroll. Это неопределённость происхождения, не само по себе нарушение покера.'),
    'MATH_DELTA': ('Деньги', 'Сумма вкладов за вычетом возвратов не равна выплатам. Возможны повтор, пропуск, неверное число или принадлежность сообщения; delta не локализует виновную строку.'),
    'MISSING_ACTION': ('Нераспознанное сообщение', 'Assembled message остаётся UNKNOWN_EVENT либо имеет неподдерживаемый action. Это может быть действие, карточный OCR, split continuation или информационный шум; пропуск реального хода не доказан одним этим кодом.'),
    'MISSING_AMOUNT': ('Сумма отсутствует или неприемлема', 'Для денежного события amount не разбирается в конечный Decimal>0. Такая запись не добавляется в ledger. Проверить оригинальный текст и continuation; не подставлять число.'),
    'CHAT_OVERLAP_DUPLICATE': ('Повтор или неоднозначная граница', 'Уже встречался forced contribution/return/street либо NEW_HAND появился повторно или после начала ставок. Код выявляет повтор в модели, но не доказывает stale duplicate. Повтор одного explicit hand ID имеет WARNING и в этот inventory не входит.'),
    'STALE_BLIND_OR_ANTE': ('Forced contribution в неверном месте', 'Ante/blind наблюдается после начала betting, после PREFLOP либо после выплаты. Возможны повтор старой строки или неверная граница окна; источник проверяется отдельно.'),
    'BETTING_ROUND_INCOMPLETE': ('Раунд не закрыт', 'При переходе улицы/showdown/завершении остаётся ACTIVE игрок, не уравнявший bet или не выполнивший ответ. Неверное имя, пропущенный fold/call и повторный raise могут вызвать этот код.'),
    'OLD_WIN': ('Выплата до завершения', 'В модели несколько live игроков, а раунд не закрыт либо нет RIVER и не все live ALL_IN. Название OLD_WIN не является доказательством старой выплаты.'),
    'CARD_EVIDENCE_MISSING': ('Не хватает карточного evidence', 'Нет полного board/hole cards для проверки, отсутствует matching physical evidence или нельзя проверить описанный showdown class. Это NOT_VERIFIABLE, не наблюдаемое карточное противоречие.'),
    'CARD_LOGIC_CONFLICT': ('Карточное противоречие', 'Наблюдаемые карты/их binding/showdown или payout противоречат модели: duplicate/shared card, board prefix, folded SHOW, repeated SHOW mismatch, class mismatch, wrong winner. Конкретный subtype указан в detail.'),
    'SIDE_POT_NOT_VERIFIABLE': ('Eligibility не доказана', 'Ledger неполон/неопределён либо нет карт для сравнения winners. Иногда модель видит payout выше ceiling, но из-за неполноты не утверждает конфликт. Это downstream uncertainty.'),
    'UNCALLED_RETURN_NOT_VERIFIABLE': ('Возврат не доказан', 'В модели есть единственный максимальный net contribution, но ledger неполон. Отсутствующий возврат не доказан: unmatched layer может возникнуть из пропуска или повтора.'),
    'MISSING_UNCALLED_RETURN': ('Неучтённый возврат', 'На доказанном ledger единственный максимальный net создаёт unmatched contribution layer, которая не должна распределяться как pot.'),
    'SIDE_POT_ELIGIBILITY_CONFLICT': ('Eligibility противоречит ledger', 'При доказанной полноте ledger payout превышает eligible ceiling либо pot layer не имеет eligible игрока. Это отличается от SIDE_POT_NOT_VERIFIABLE.'),
    'INCOMPLETE_START': ('Неполное начало', 'Нет наблюдаемого start marker либо в окне отсутствует распознанная пара SB/BB. Частичный хвост и split blind могут давать этот код.'),
    'INCOMPLETE_END': ('Неполное завершение', 'Нет наблюдаемой выплаты/конца либо HAND_END не имеет legal round closure/payout. Код относится к полноте окна.'),
    'HAND_BOUNDARY_CONFLICT': ('Граница руки', 'В окне противоречат numbered markers или следующий numbered hand пришёл до наблюдаемого завершения предыдущего. Проверить source chronology, не объединять окна автоматически.'),
    'PLAYER_IDENTITY_CONFLICT': ('Имя игрока', 'Имя отсутствует/содержит ? или replacement character либо есть неподтверждённый punctuation alias. Похожие другие OCR spellings тоже могут разнести одного игрока по нескольким состояниям без этого кода.'),
    'INVALID_CALL': ('Некорректный call в модели', 'При CALL=ADDED требуемая добавка=bet−street_contribution. Внесённая сумма должна быть положительной и не выше required; меньшая допустима только с observed all-in.'),
    'CALL_WITH_NOTHING_TO_CALL': ('Call без долга', 'bet−street_contribution<=0 перед CALL. Возможны повтор call, неверный street reset из грязного потока либо противоречивый action.'),
    'INVALID_RAISE': ('Некорректный raise-to в модели', 'До raise должен быть bet>0; target должен быть выше bet и выше собственного street contribution. Проверка minimum legal raise в этот код не включена.'),
    'BET_INTO_EXISTING_BET': ('Bet при существующей ставке', 'Перед BET current bet!=0. Нужен raise либо источник содержит повтор/ошибочный action.'),
    'CHECK_FACING_BET': ('Check при долге', 'Перед CHECK current bet отличается от вклада игрока на улице.'),
    'ACTION_AFTER_FOLD': ('Действие после fold', 'Перед betting action игрок уже FOLDED. Повторный fold тоже вызывает код. Это нарушение observed sequence; реальная игра могла быть корректна при stale row.'),
    'ACTION_AFTER_ALLIN': ('Действие после all-in', 'Перед betting action игрок уже ALL_IN. Проверить observed all-in continuation и повтор/границу.'),
    'ACTION_AFTER_HAND_END': ('Действие после выплаты', 'Betting action встретился после ended=True, которое выставляется на WIN/HAND_END. Возможны stale win или boundary bleed.'),
    'BROKEN_STREET_ORDER': ('Порядок улиц', 'Новая улица не следующая за текущей либо появляется после payout. Пропуск board observation и повторная улица требуют проверки источника.'),
    'RETURN_EXCEEDS_CONTRIBUTION': ('Возврат выше внесённого', 'Возврат превышает вклад на улице или остаток total−предыдущие returns; math дополнительно проверяет negative net. Может сработать дважды: локально и на итоговом ledger.'),
    'INVALID_UNCALLED_RETURN': ('Не совпадает unmatched return', 'Возврат должен равняться own street contribution−max(other street contributions) в текущей модели. Загрязнение ledger может сделать эту проверку каскадной.'),
    'INELIGIBLE_PAYOUT': ('Выплата folded игроку', 'Перед WIN состояние получателя=FOLDED. Требуется проверка имен/повтора/границы, а не автоматическое восстановление игрока.'),
    'ODD_CHIP_ALLOCATION_UNPROVEN': ('Распределение нечётных фишек', 'Равное деление pot layer даёт нецелую share; не доказан порядок назначения odd chip.'),
    'AMOUNT_SEMANTICS_UNPROVEN': ('Не доказана семантика суммы', 'CALL/BET contract отсутствует или неизвестен. Для данного replay сохранён доказанный ADDED contract.'),
}


def safe(value):
    return str(value if value is not None else '—').replace('|', '\\|').replace('\n', ' ')


def link(path):
    return '<'+path.resolve().as_posix()+'>'


def context(record, hand):
    code = record['issue']['code']
    before = record.get('BEFORE', record.get('before')) or {}
    msg = record.get('message') or {}
    amount = msg.get('amount')
    bet = before.get('current_bet')
    contribution = before.get('player_street_contribution')
    bits = []
    if before:
        bits.append(f"street={before.get('street')}; bet={bet}; player={before.get('player')}; state={before.get('player_state')}; contribution={contribution}")
    if amount is not None:
        bits.append('parsed amount='+str(amount))
    if code in {'INVALID_CALL', 'CALL_WITH_NOTHING_TO_CALL', 'CHECK_FACING_BET'} and bet is not None and contribution is not None:
        bits.append('required='+str(Decimal(str(bet))-Decimal(str(contribution))))
    if code == 'BETTING_ROUND_INCOMPLETE' and before:
        pending = [f"{n}:contribution={p['street_contribution']},acted={p['acted']}" for n, p in before.get('players', {}).items()
                   if p['state'] == 'ACTIVE' and (str(p['street_contribution']) != str(bet) or not p['acted'])]
        bits.append('ACTIVE pending: '+', '.join(pending))
    if code == 'MATH_DELTA':
        bits.append(f"net={hand.get('math_net_total')}; payout={hand.get('math_payout_total', hand.get('payout_total'))}; delta={hand['math_delta']}")
    return '; '.join(bits)


def main():
    summary = json.loads((BASE/'summary.json').read_text(encoding='utf-8'))
    records = [json.loads(s) for s in (BASE/'blockers.jsonl').read_text(encoding='utf-8').splitlines() if s]
    hands = json.loads((BASE/'handmatrix.json').read_text(encoding='utf-8'))
    lookup = {(h['table_id'], h['hand_epoch']): h for h in hands}
    byhand = defaultdict(list)
    bycode = defaultdict(list)
    for r in records:
        byhand[(r['table_id'], r['hand_epoch'])].append(r)
        bycode[r['issue']['code']].append(r)
    count = lambda rs: len({(r['table_id'], r['hand_epoch']) for r in rs})
    ordered = sorted(bycode, key=lambda c: (-count(bycode[c]), c))
    detail = ['# Smoke17: все текущие BLOCKER по рукам', '',
              'Фиксируется текущий результат диагностики. Строки не удалены, исправления не внесены. Если источник не позволяет доказать первопричину, это указано как неопределённость. Для message_index=-1 проблема относится к итоговой модели всей руки: конкретная виновная source row не установлена.', '']
    for key, hand in lookup.items():
        rs = byhand.get(key, [])
        detail += [f"## {hand['table_id']} / hand {hand['hand_id'] or 'UNKNOWN'} / epoch {hand['hand_epoch']}", '',
                   f"Final={hand['status']}; action={hand['action_logic_status']}; cards={hand['card_status']}; sidepot={hand['side_pot_status']}; identity={hand['identity_status']}; math delta={hand['math_delta']}; blockers={len(rs)}.", '']
        if not rs:
            detail += ['BLOCKER отсутствуют.', '']
            continue
        detail += ['| Код | Где | Исходный текст | Почему / наблюдаемое |', '|---|---|---|---|']
        for r in rs:
            issue = r['issue']
            msg = r.get('message')
            where = (', '.join(issue['source_row_ids'])+f"; t={msg.get('timestamp_ms')}ms; frame={msg.get('frame_id')}; message={issue['message_index']}" if msg
                     else 'Вся рука; message_index=-1; локальная source row не установлена')
            reason = RULES.get(issue['code'], ('', 'Condition см. issue detail'))[1]
            observed = context(r, hand)
            explain = reason+' Detail: '+str(issue.get('detail') or '—')
            if observed:
                explain += ' Наблюдаемое: '+observed
            detail.append('| '+ ' | '.join(map(safe, [issue['code'], where, msg.get('raw_text') if msg else '—', explain]))+' |')
        detail.append('')
    detail_path = BASE/'errors_by_hand.md'
    detail_path.write_text('\n'.join(detail)+'\n', encoding='utf-8')
    statuses = Counter(h['status'] for h in hands)
    lines = ['# Текущие блокеры Smoke17', '',
             'Проверен текущий экспериментальный Logic Engine на сохранённом Smoke17:199 candidate windows. Видео не декодировалось, Jev не вызывался, движки/Excel/source rows не изменялись. Набор не является очищенным Excel. Все результаты повторного validate_hand сопоставлены с сохранённым final replay; receipt приведён ниже.', '',
             f"Рук с BLOCKER: {len(byhand)}/199. Рук без BLOCKER: {199-len(byhand)}. Срабатываний BLOCKER: {len(records)}. Различных встречающихся кодов: {len(bycode)}. Частоты ниже — число рук с кодом, они пересекаются и не являются независимыми root causes.", '',
             '## Какие блокеры и почему', '', '| Код | Рук | Срабатываний | Что это означает сейчас |', '|---|---:|---:|---|']
    for code in ordered:
        lines.append(f"| {code} | {count(bycode[code])} | {len(bycode[code])} | {RULES.get(code, ('', 'См. detail'))[1]} |")
    zero = [h for h in hands if h['math_delta'] == '0']
    lines += ['', '## Почему нельзя считать все коды доказанными ошибками игры', '',
              '1. ROW_IDENTITY_UNRESOLVED означает неопределённую идентичность наблюдения при scroll. Он не доказывает неверную ставку. Harmless informational rows имеют WARNING и исключены из статистики BLOCKER.',
              '2. MISSING_ACTION означает UNKNOWN assembled message. Среди них могут остаться server notices, continuation fragments и карточный OCR. Название кода не доказывает отсутствующий poker action.',
              '3. CHAT_OVERLAP_DUPLICATE, OLD_WIN и STALE_BLIND_OR_ANTE — названия условий модели. Они не являются независимым доказательством stale: нельзя удалять строку только по такому коду.',
              '4. Отсутствие суммы/действия и повтор денежных строк могут изменить player state/ledger. Затем BETTING_ROUND_INCOMPLETE, OLD_WIN, INVALID_CALL, RETURN и MATH_DELTA возникают каскадом. Простое сложение их counts завысит число первопричин.',
              '5. CARD_EVIDENCE_MISSING, SIDE_POT_NOT_VERIFIABLE и UNCALLED_RETURN_NOT_VERIFIABLE означают недостаток evidence. Они продолжают входить в certification_blockers, хотя не доказывают противоречие.',
              '6. INVALID_CALL/RAISE/CHECK/AFTER_FOLD сравнивают parsed action с текущим modeled state. Это доказанное противоречие внутри данного observed stream; источник реальной игры может быть корректным при грязном потоке.',
              '7. evidence_type=CONTRADICTION — метка текущей реализации, не независимый вердикт. Например, некоторые missing completion/unknown cases получают её по default. domain=ACTION у некоторых returns и отсутствие единой локальной строки для итоговых ошибок также являются особенностями текущего контракта.', '',
              '## Контрольный пул math delta=0', '',
              f"Фиксированный пул содержит {len(zero)} окна. Delta0 означает только равенство наблюдаемого ledger выплатам; часть окон неполная. Action statuses: {dict(Counter(h['action_logic_status'] for h in zero))}. Final statuses: {dict(Counter(h['status'] for h in zero))}.", '',
              '| Стол | Hand / epoch | Delta | Action | Cards | Sidepot | BLOCKER codes |', '|---|---|---:|---|---|---|---|']
    for h in zero:
        rs = byhand.get((h['table_id'], h['hand_epoch']), [])
        lines.append('| '+ ' | '.join(map(safe, [h['table_id'], f"{h['hand_id'] or 'UNKNOWN'} /{h['hand_epoch']}", h['math_delta'], h['action_logic_status'], h['card_status'], h['side_pot_status'], ', '.join(sorted({r['issue']['code'] for r in rs})) or 'нет']))+' |')
    lines += ['', '## Примеры с конкретными исходными строками', '']
    for code in ordered:
        r = next((r for r in bycode[code] if r.get('message')), bycode[code][0])
        msg = r.get('message')
        h = lookup[(r['table_id'], r['hand_epoch'])]
        lines += [f"### {code}: {r['table_id']} / {r['hand_id'] or 'UNKNOWN'} / epoch{r['hand_epoch']}", '',
                  'Source rows: '+(', '.join(r['issue']['source_row_ids']) or 'итог всей руки; конкретная source row не установлена')+'.',
                  ('Timestamp='+str(msg.get('timestamp_ms'))+'ms; frame='+str(msg.get('frame_id'))+'.' if msg else ''),
                  ('Raw text: `'+safe(msg['raw_text']).replace('`', '\\`')+'`' if msg else ''),
                  'Почему: '+RULES.get(code, ('', 'См. detail'))[1],
                  'Detail: '+str(r['issue'].get('detail') or '—'),
                  'Наблюдаемое: '+(context(r, h) or 'см. состояние/ledger всей руки в machine-readable inventory'), '']
    lines += ['## Итоговые статусы и приоритет проверки источников', '', '| Final status | Рук |', '|---|---:|']
    lines += [f'| {status} | {n} |' for status, n in sorted(statuses.items())]
    lines += ['', 'Приоритет очистки: сначала доказать observation duplicates и проверить hand boundary/многострочные continuations; затем разбирать суммы/имена/карточные связи. После этого повторить диагностику и сравнить исчезнувшие и сохранившиеся коды. Подозрительные строки без независимого доказательства остаются AMBIGUOUS. В этом этапе ничего не очищалось и не исправлялось.', '',
              f"[Все проблемы по каждой руке]({link(detail_path)}): код, стол, hand/epoch, source rows, время, raw text, конкретное условие и наблюдаемый state. [Полный JSONL]({link(BASE/'blockers.jsonl')}); [матрица199 рук]({link(BASE/'handmatrix.json')}); [receipt проверок и hashes]({link(BASE/'summary.json')}).", '',
              '## Проверка воспроизводимости', '', '```json', json.dumps(summary, ensure_ascii=False, indent=2), '```', '',
              'Код production/движков не менялся. HH не экспортировались. Формат BLOCKER здесь объясняет, почему текущий engine не поставил FULLY_VALID_HAND; обработка сохранённого набора прошла до конца.']
    report = ROOT/'docs/CURRENT_BLOCKERS_SMOKE17_20261001.md'
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'report': str(report), 'detail': str(detail_path), 'hands': len(hands),
                      'hands_with_blockers': len(byhand), 'blocker_occurrences': len(records),
                      'codes': {c: {'hands': count(bycode[c]), 'occurrences': len(bycode[c])} for c in ordered}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
