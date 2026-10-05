"""Нормализация полей реестра: ключ субагента, билеты, суммы, пассажиры.

Ключ субагента и билеты строятся так же, как в cleaning.py (P2).
"""
import re
from dataclasses import dataclass

CURRENCIES = frozenset({"KGS", "USD", "EUR", "RUB", "KZT"})


class ParseError(ValueError):
    """Ошибка разбора ячейки, reason идёт в parse_status."""

    def __init__(self, reason, detail=""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# ================================================================
# ЧАСТЬ 1. КОНТРАГЕНТЫ И БИЛЕТЫ
# ================================================================

# Шаблоны для юридических форм и кавычек (как в norm_name у P2)
_LEGAL_FORM = re.compile(r"\b(осоо|ооо|ип|зао|оао)\b", re.I)
_QUOTES = re.compile(r"[«»\"“”'`]")


def subagent_key(name) -> str:
    """Приводит название субагента к единому ключу, как norm_name в cleaning.py."""
    if name is None:
        return ""
    s = str(name).lower().replace("ё", "е")
    s = _QUOTES.sub("", s)
    s = _LEGAL_FORM.sub(" ", s)
    return re.sub(r"\s+", "", s)


# Шаблоны для номеров билетов
_TICKET_SEP = re.compile(r"[\s,;/|]+")
_TICKET = re.compile(r"\d{3}-\d{10}|\d{13}|\d{10}")


def tickets10(cell) -> list:
    """Берёт последние 10 цифр каждого билета из ячейки."""
    if cell is None:
        raise ParseError("no_tickets")
    parts = [p for p in _TICKET_SEP.split(str(cell)) if p]
    if not parts:
        raise ParseError("no_tickets")
    # Любой кусок, который не билет, считаем ошибкой, а не пропускаем
    bad = [p for p in parts if not _TICKET.fullmatch(p)]
    if bad:
        raise ParseError("bad_ticket_format", ", ".join(bad))
    result = [p.replace("-", "")[-10:] for p in parts]
    if len(set(result)) != len(result):
        raise ParseError("duplicate_ticket_in_cell", str(cell))
    return result


# ================================================================
# ЧАСТЬ 2. ЧИСЛА И СУММЫ
# ================================================================

_SPACES = re.compile(r"[ \xa0\u202f]")


def parse_number(text) -> float:
    """Превращает число из реестра (7.889,70 / 115 877,00 / 20062) в float."""
    raw = str(text)
    s = _SPACES.sub(" ", raw).strip()
    negative = s.startswith("-")
    if negative:
        s = s[1:].strip()
    if not s or not re.fullmatch(r"\d[\d .,]*", s) or not s[-1].isdigit():
        raise ParseError("bad_number", raw)

    # Десятичным считаем разделитель, который стоит правее
    has_dot, has_comma = "." in s, "," in s
    if has_dot and has_comma:
        dec = "." if s.rfind(".") > s.rfind(",") else ","
        thou = "," if dec == "." else "."
    elif has_comma:
        if s.count(",") > 1:
            raise ParseError("bad_number", raw)
        dec, thou = ",", None
    elif has_dot:
        dec, thou = (".", None) if s.count(".") == 1 else (None, ".")
    else:
        dec = thou = None

    int_part, frac = s.rsplit(dec, 1) if dec else (s, "")
    if dec and (dec in int_part or not frac.isdigit()):
        raise ParseError("bad_number", raw)
    # Одиночное 12.345 может быть и тысячами, и дробью, поэтому не угадываем
    if dec and thou is None and " " not in int_part and len(frac) == 3:
        raise ParseError("ambiguous_number", raw)

    # Группы тысяч должны быть по 3 цифры
    groups = re.split(r"[ " + re.escape(thou) + "]" if thou else " ", int_part)
    if not all(g.isdigit() for g in groups):
        raise ParseError("bad_number", raw)
    if len(groups) > 1 and (len(groups[0]) > 3 or any(len(g) != 3 for g in groups[1:])):
        raise ParseError("bad_number", raw)

    value = float("".join(groups) + ("." + frac if frac else ""))
    return -value if negative else value


# Шаблоны для ячейки суммы (pay_cell)
_NUM = r"-?\s*\d[\d .,\xa0\u202f]*\d|-?\s*\d"
_CUR = r"[a-z]{3}"
_MAIN = re.compile(rf"(?:(?P<c1>{_CUR})\s*(?P<n1>{_NUM})|(?P<n2>{_NUM})\s*(?P<c2>{_CUR}))")
_FEE = re.compile(rf"(?:(?P<c1>{_CUR})\s*(?P<n1>{_NUM})|(?P<n2>{_NUM})\s*(?P<c2>{_CUR}))\s*sf")
_PENALTY = re.compile(r"\(\s*штраф\s*(?P<pct>\d+(?:[.,]\d+)?)\s*%\s*\)")
_BARE_TICKET = re.compile(r"\d{3}-?\d{10}|\d{10}")


@dataclass
class Pay:
    amount: float
    currency: str
    fee: float = None
    fee_currency: str = None
    penalty_pct: float = None


def _currency(code) -> str:
    cur = code.upper()
    if cur not in CURRENCIES:
        raise ParseError("unknown_currency", code)
    return cur


def parse_pay(cell) -> Pay:
    """Разбирает ячейку суммы: основная сумма, сбор (sf) и штраф."""
    raw = "" if cell is None else str(cell)
    low = _SPACES.sub(" ", raw).lower().strip()
    if not low:
        raise ParseError("no_amount")

    # Вырезаем штраф в скобках
    penalty = None
    pen = list(_PENALTY.finditer(low))
    if len(pen) > 1:
        raise ParseError("bad_amount_format", raw)
    if pen:
        penalty = float(pen[0].group("pct").replace(",", "."))
        low = (low[: pen[0].start()] + low[pen[0].end():]).strip()
    if "(" in low or ")" in low:
        raise ParseError("bad_amount_format", raw)

    # Делим на основную сумму и сбор
    parts = low.split("+")
    if len(parts) > 2:
        raise ParseError("bad_amount_format", raw)
    main = parts[0].strip()
    fee_text = parts[1].strip() if len(parts) == 2 else None

    m = _MAIN.fullmatch(main)
    if not m:
        if _BARE_TICKET.fullmatch(main):
            raise ParseError("ticket_in_amount", raw)
        if re.fullmatch(_NUM, main):
            raise ParseError("no_currency", raw)
        raise ParseError("bad_amount_format", raw)
    currency = _currency(m.group("c1") or m.group("c2"))
    amount = parse_number(m.group("n1") or m.group("n2"))

    fee = fee_currency = None
    if fee_text is not None:
        f = _FEE.fullmatch(fee_text)
        if not f:
            raise ParseError("bad_fee", raw)
        fee_currency = _currency(f.group("c1") or f.group("c2"))
        fee = parse_number(f.group("n1") or f.group("n2"))
        if fee < 0:
            raise ParseError("bad_fee", raw)

    return Pay(amount, currency, fee, fee_currency, penalty)


# ================================================================
# ЧАСТЬ 3. ПАССАЖИРЫ
# ================================================================

def normalize_pax(cell) -> list:
    """Приводит список пассажиров к верхнему регистру без лишних пробелов."""
    if cell is None:
        return []
    names = re.split(r"[,;\n]+", str(cell))
    return [re.sub(r"\s+", " ", n).strip().upper() for n in names if n.strip()]
