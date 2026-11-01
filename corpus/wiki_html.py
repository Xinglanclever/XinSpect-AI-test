#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Wikipedia 的 **渲染後 HTML** 拆成表格列。

為什麼還需要這一條（wikitext 已經有了）：
  有些頁面的型號表不是寫在頁面裡，而是用模板轉嵌進來的
  （`{{AMD Radeon RX 6000 series}}`、`{{cpulist}}` 的渲染結果、`{{#invoke:}}`）。
  這些在 wikitext 裡只看得到模板名，看不到內容；渲染後的 HTML 才有真正的表格。

輸出格式與 import_wiki_wikitext.parse_table 一致：list[table] → list[row] → list[cell]，
rowspan／colspan 一樣展開，所以兩邊可以共用同一套型號抽取邏輯。

只用標準庫（html.parser），不引入 bs4/lxml——這支要能在使用者的機器上直接跑。
"""

import html as _html
import re
from html.parser import HTMLParser

CELL_TAGS = {"td", "th"}
SKIP_TAGS = {"script", "style", "sup"}          # 註腳編號、樣式對型號沒用


def _squash(s):
    s = _html.unescape(s or "")
    s = s.replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


class _TableParser(HTMLParser):
    """把每一張表（含巢狀表）拆成列。

    巢狀表格要各自收集：Radeon RX 7000 這種頁面把每個區隔（Enthusiast／High-end…）
    做成一張內層表，真正的規格列在內層。只取最外層的話，只會拿到
    「Enthusiast | Radeon RX 7900 XTX」這種包裝列，顯存那一欄整批不見。
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []          # list[list[list[str]]]
        self._frames = []         # 每層表格：rows / row / col / pending
        self._cell = None
        self._cell_frame = None
        self._cell_attrs = (1, 1)
        self._skip_depth = 0

    # ── 表格邊界 ────────────────────────────────────────────────
    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "table":
            self._frames.append({"rows": [], "row": None, "col": 0, "pending": {}})
            return
        if not self._frames:
            return
        a = dict(attrs)
        if tag == "tr":
            self._flush_row()
        elif tag in CELL_TAGS:
            self._flush_cell()
            self._cell = []
            self._cell_frame = self._frames[-1]
            self._cell_attrs = (_int(a.get("rowspan")), _int(a.get("colspan")))
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "table":
            if not self._frames:
                return
            self._flush_row()
            f = self._frames.pop()
            if f["rows"]:
                self.tables.append(f["rows"])
            return
        if not self._frames:
            return
        if tag in CELL_TAGS:
            self._flush_cell()

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._cell is not None:            # 儲存格外的文字（空白、分隔符）不需要
            self._cell.append(data)

    # ── 列／儲存格 ──────────────────────────────────────────────
    def _flush_cell(self):
        if self._cell is None:
            return
        text = _squash("".join(self._cell))
        frame = self._cell_frame
        self._cell = None
        self._cell_frame = None
        if frame is None:
            return
        if frame["row"] is None:
            frame["row"] = []
            frame["col"] = 0
        col = frame["col"]
        while col in frame["pending"]:
            v, left = frame["pending"][col]
            frame["row"].append(v)
            if left <= 1:
                del frame["pending"][col]
            else:
                frame["pending"][col] = [v, left - 1]
            col += 1
        rs, cs = self._cell_attrs
        frame["row"].append(text)
        if rs > 1:
            frame["pending"][col] = [text, rs - 1]
        # colspan 跨過的欄位要清掉殘留的 rowspan，否則那格的值會一路帶進後面的列。
        for _ in range(1, max(1, cs)):
            col += 1
            frame["pending"].pop(col, None)
        frame["col"] = col + 1

    def _flush_row(self):
        if not self._frames:
            return
        self._flush_cell()
        frame = self._frames[-1]
        if frame["row"]:
            frame["rows"].append(frame["row"])
        frame["row"] = None
        frame["col"] = 0


def _int(v, default=1):
    try:
        n = int(str(v).strip())
        return n if n > 0 else default
    except (TypeError, ValueError):
        return default


def parse_tables(html_text):
    """回傳 list[table]；table 是 list[row]；row 是 list[str]。"""
    p = _TableParser()
    try:
        p.feed(html_text)
        p.close()
    except Exception:                        # noqa: BLE001 —— HTML 壞掉不該讓整批停
        pass
    return p.tables
