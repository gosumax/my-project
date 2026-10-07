"""Read workbook evidence and append a column without rewriting source cells."""
from __future__ import annotations

import posixpath
import re
import zipfile
from xml.etree import ElementTree as ET

NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def column_number(address):
    n = 0
    for char in re.match(r"[A-Z]+", address)[0]:
        n = n * 26 + ord(char) - 64
    return n


def column_name(n):
    text = ""
    while n:
        n, digit = divmod(n-1, 26)
        text = chr(65+digit) + text
    return text


def read_workbook(path):
    sheets = {}
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(si.itertext()) for si in root]
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {r.get("Id"): r.get("Target") for r in rels}
        root = ET.fromstring(archive.read("xl/workbook.xml"))
        for sheet in root.find(f"{{{NS}}}sheets"):
            target = targets[sheet.get(f"{{{REL}}}id")]
            target = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/"+target)
            xml = ET.fromstring(archive.read(target))
            grid = []
            cells = {}
            for row in xml.find(f"{{{NS}}}sheetData"):
                values = {}
                for cell in row:
                    address = cell.get("r")
                    cells[address] = ET.tostring(cell, encoding="unicode")
                    value = cell.find(f"{{{NS}}}v")
                    kind = cell.get("t")
                    if kind == "inlineStr":
                        value = "".join(cell.find(f"{{{NS}}}is").itertext())
                    elif value is None:
                        value = None
                    elif kind == "s":
                        value = shared[int(value.text)]
                    elif kind == "b":
                        value = value.text == "1"
                    elif kind in {"str", "e"}:
                        value = value.text
                    else:
                        value = float(value.text)
                        if value.is_integer():
                            value = int(value)
                    values[column_number(address)] = value
                grid.append((int(row.get("r")), values))
            header = grid[0][1]
            width = max(header)
            headers = [header.get(i) for i in range(1, width+1)]
            rows = [{key: values.get(i) for i, key in enumerate(headers, 1)}
                    for _, values in grid[1:]]
            sheets[sheet.get("name")] = {"part": target, "headers": headers,
                                        "rows": rows, "cells": cells,
                                        "excel_rows": [r for r, _ in grid[1:]]}
    return sheets


def append_authored_edge_column(source, donor, destination, specs):
    """Transplant Artifact Tool's authored values, preserve every old XML cell.

    The artifact round-trip is a donor only: existing native parts, styles,
    annotations, formulas and cached source values stay in the original package.
    """
    original = read_workbook(source)
    authored = read_workbook(donor)
    replacements = {}
    with zipfile.ZipFile(source) as archive:
        for name, spec in specs.items():
            sheet = original[name]
            edge_col = column_name(len(sheet["headers"])+1)
            donor_sheet = authored[name]
            actual = [r["edge_status"] for r in donor_sheet["rows"]]
            if actual != spec["values"]:
                raise ValueError(f"Authored status mismatch in {name}")
            xml = archive.read(sheet["part"]).decode("utf-8")
            prefix = re.search(r"<([\w]+:)?worksheet\b", xml).group(1) or ""
            status_by_row = dict(zip(sheet["excel_rows"], actual))
            status_by_row[1] = "edge_status"
            def patch_row(match):
                opening, content, closing = match.group(1, 2, 3)
                row_n = int(re.search(r'\br="(\d+)"', opening).group(1))
                if row_n not in status_by_row:
                    return match.group(0)
                # Reuse this row's last existing cell style; no source style edits.
                styles = re.findall(r'<'+prefix+r'c\b[^>]*\bs="(\d+)"', content)
                style = f' s="{styles[-1]}"' if styles else ""
                cell = (f'<{prefix}c r="{edge_col}{row_n}"{style} t="inlineStr">'
                        f'<{prefix}is><{prefix}t>{status_by_row[row_n]}</{prefix}t>'
                        f'</{prefix}is></{prefix}c>')
                return opening+content+cell+closing
            xml, count = re.subn(r'(<'+prefix+r'row\b[^>]*>)(.*?)(</'+prefix+r'row>)',
                                patch_row, xml, flags=re.S)
            if count < len(status_by_row):
                raise ValueError(f"Missing physical rows in {name}")
            col = len(sheet["headers"])+1
            column = f'<{prefix}col min="{col}" max="{col}" width="23" customWidth="1" />'
            if f'</{prefix}cols>' in xml:
                xml = xml.replace(f'</{prefix}cols>', column+f'</{prefix}cols>', 1)
            else:
                xml = xml.replace(f'<{prefix}sheetData>', f'<{prefix}cols>'+column+f'</{prefix}cols><{prefix}sheetData>', 1)
            xml = re.sub(r'(<'+prefix+r'dimension\b[^>]*\bref=")[^"]+("[^>]*/>)',
                         lambda m: m[1]+f'A1:{edge_col}{max(status_by_row)}'+m[2], xml)
            replacements[sheet["part"]] = xml.encode("utf-8")
        with zipfile.ZipFile(destination, "x") as output:
            for info in archive.infolist():
                output.writestr(info, replacements.get(info.filename, archive.read(info.filename)))
    after = read_workbook(destination)
    parity = {}
    for name, before in original.items():
        new = after[name]
        if name in specs:
            if len(before["rows"]) != len(new["rows"]) or new["headers"][:-1] != before["headers"]:
                raise ValueError(f"Row/header preservation failed: {name}")
            if any(new["cells"].get(address) != value for address, value in before["cells"].items()):
                raise ValueError(f"Original cell preservation failed: {name}")
            if [r["edge_status"] for r in new["rows"]] != specs[name]["values"]:
                raise ValueError(f"Exported status mismatch: {name}")
        elif before != new:
            raise ValueError(f"Untouched sheet changed: {name}")
        parity[name] = {"before_rows": len(before["rows"]), "after_rows": len(new["rows"]),
                        "original_cells_unchanged": True}
    with zipfile.ZipFile(source) as a, zipfile.ZipFile(destination) as b:
        untouched = [p for p in a.namelist() if p not in replacements]
        if any(a.read(p) != b.read(p) for p in untouched):
            raise ValueError("Untouched XLSX part changed")
    return {"sheets": parity, "untouched_parts_byte_equal": len(untouched)}
