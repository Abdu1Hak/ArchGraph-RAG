"""Strict, lossless-meaning normalizers for extracted PDF passages, section paths, and generated answers.

Handles PyMuPDF4LLM extraction artifacts (`<!--...-->`, `<br>`, `<sup><u>...</u></sup>`,
`<u>...</u>`, prose-swallowing `<sup>...</sup>`, `~~...~~`, `\ufffd` glyphs, control chars,
glued underscore italics like `_vk_and_σ_2=`, and broken OCR spacing) while strictly preserving
valid `snake_case` identifiers, code blocks, URLs, `[chunk_id]` citations, and token tags
(`<eos>`, `<bos>`, `<pad>`, `<resv>`).
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

from src.generation.citation_validator import CITATION_RE

_SENTINEL_LEFT = "\ue000"
_SENTINEL_RIGHT = "\ue001"

_HTML_COMMENT_RE = re.compile(r"<!--[\s\S]*?-->")
_BR_TAG_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_CTRL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INVISIBLES_RE = re.compile(r"[\ufe00-\ufe0f\u200b\ufeff]")

# Token tags and literal structural tags that must be preserved verbatim.
_TOKEN_TAG_RE = re.compile(
    r"</?(?:eos|bos|pad|resv|unk|sep|cls|mask|start_of_turn|end_of_turn|think|answer)>",
    re.IGNORECASE,
)
_EMAIL_ANGLE_RE = re.compile(r"<_?\s*([A-Za-z0-9_.+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\s*_?>")
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```|`[^`\n]+`")
_URL_RE = re.compile(r"https?://[^\s<>\"'`\])]+")
_EINSUM_STR_RE = re.compile(r'"[^"\n]{1,80}(?:->|\u2212>)[^"\n]{0,80}"')

# Valid snake_case IDs & domain identifiers that must NEVER have underscores unwrapped.
_VALID_SNAKE_CASE_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    r"[a-z]{2,}(?:_[a-z0-9]+)+"
    r"|d_(?:model|ff|attn|embd|k|v|q|h|c|cl)"
    r"|n_(?:layer|vocab|ctx)"
    r"|r_(?:min|max)"
    r"|(?:prev|new)_[KV]"
    r"|P_[qkvo]"
    r"|[A-Za-z0-9]+(?:_[a-z0-9]+){2,}"
    r")(?![A-Za-z0-9_])"
)

# Section path cleanup regexes (backward-compatible with existing tests).
_SEC_BOLD_RE = re.compile(r"\*\*([^*\n]*?)\*\*")
_SEC_EMPH_UNDER_RE = re.compile(r"(?<!\w)_([^_\n]+?)_(?!\w)")
_SEC_EMPH_STAR_RE = re.compile(r"(?<!\w)\*([^*\n]+?)\*(?!\w)")
_SEC_STRAY_STAR_RE = re.compile(r"(?<!\w)\*(?!\s)|(?<!\s)\*(?!\w)")

# Strikethrough/overbar artifacts around square roots.
_TILDE_SQRT_RE = re.compile(r"~~\s*(?:_?\s*)?(?:<u>\s*)?[\ufffd\u221a](?:\s*</u>)?(?:\s*_?)?\s*~~")
_TILDE_WRAP_RE = re.compile(r"~~([^~\n]{1,32})~~")

# Specific PDF fraction/underline constructs observed across the 35-paper corpus.
_FRAC_NUM_SPLIT_RE = re.compile(
    r"<sup>\s*_?([^<>\n]{1,18}?)[-\u2212]_?\s*</sup>\s*_?([A-Za-z\u0370-\u03ff]{1,6})_?\s*"
    r"<sup>\s*_?<u>([^<>\n]{1,18}?)</u>_?\s*</sup>"
)
_FRAC_JACOBIAN_RE = re.compile(
    r"<sup>\s*_?\u2202_?\s*</sup>\s*<sup>\s*<u>([^<>\n]{1,12})</u>\s*</sup>\s*"
    r"_?\u2202([A-Za-z]{1,6})_?\s*<sup>\s*<u>\(</u>\s*</sup>\s*<sup>\s*_?([A-Za-z0-9]+)_?\s*</sup>\s*"
    r"<sup>\s*<u>\)</u>\s*</sup>"
)
_FRAC_DEN_SUP_U_RE = re.compile(
    r"(?<![A-Za-z0-9_])_?(d|n|k|B|N|\u03b2|r|s|L|\u03c3)_?\s*"
    r"<sup>\s*_?<u>([A-Za-z0-9\u0370-\u03ff\u0394\u2206*\u2032+\-]+)</u>_?\s*</sup>"
    r"(?:\s*_?([rc]|model|ff|attn)_?(?=\s|[.,;:)\]\u2309]|$))?"
)
_FRAC_EQ_SUP_U_RE = re.compile(
    r"(^|[\s(=+\-*\u00b7])\s*<sup>\s*_?<u>([A-Za-z0-9\u0370-\u03ff\u0394\u2206]{1,10})</u>_?\s*</sup>\s*"
    r"_?([A-Za-z0-9\u0370-\u03ff\u0394\u2206\u2032*]{1,8})_?"
    r"(?:\s*<sup>\s*_?([\u2032'*])_?\s*</sup>)?"
)
_FRAC_SQRT_UNDER_1_RE = re.compile(
    r"(?:\u221a\s*<u>\s*1\s*</u>|<u>\s*1\s*</u>\s*\u221a)\s*_?([A-Za-z0-9_\u0370-\u03ff]{1,8})_?"
)
_FRAC_SQRT_K_UNDER_1_RE = re.compile(
    r"\u221a\s*_?([kn])_?\s*<u>\s*1\s*(?:[\ufffd\u2211])?\s*</u>"
)
_UNDERLINE_TAG_RE = re.compile(r"<u>([\s\S]*?)</u>", re.IGNORECASE)

# Sum/index patterns that indicate a single U+FFFD glyph was a summation symbol (`∑`).
_MULTI_FFFD_RE = re.compile(r"\ufffd{2,}")
_FFFD_BEFORE_SUM_RE = re.compile(
    r"(?:<sup>\s*)?\ufffd(?:\s*</sup>)?(?=\s*(?:"
    r"_?[A-Za-z][ijknt]_?\s*=\s*1\b"
    r"|(?:<sup>[^<\n]{1,10}</sup>\s*)*_?[ijknt]_?\s*=\s*1\b"
    r"|\|o_?i\|"
    r"|G(?:\s|$|\s*_?[ij])"
    r"|l\s*=\s*0\b"
    r"|_?l_?\s*(?:<sup>\s*_?[-−]_?\s*</sup>\s*=0|\s*-\s*1)"
    r"|_?j_?\s*\u2208"
    r"|_?y_?(?:\s|$|<sup>)"
    r"|_?i_?(?:\s|$|<sup>)"
    r"|_?\u03b1_?i"
    r"|\u2202"
    r"|Runs\b"
    r"))"
)
_SUM_SUP_LIMIT_RE = re.compile(
    r"\u2211\s*<sup>\s*_?([A-Za-z])_?\s*</sup>\s*_?([ijknt])_?\s*=\s*1(?![0-9])"
)
_SUM_COMBINED_LIMITS_RE = re.compile(
    r"\u2211\s*_?([dknNtG])([ijknt])_?\s*=\s*1(?![0-9])"
)
_SUM_LOWER_ONLY_RE = re.compile(
    r"\u2211\s*_?([ijkntl])_?\s*=\s*([01])(?![0-9])"
)

# Short math superscript detection inside <sup>...</sup>.
_PROSE_WORDS_RE = re.compile(
    r"\b(?:the|and|for|with|where|from|that|this|these|those|which|when|while|"
    r"are|was|were|has|have|had|can|will|not|into|each|denotes|elements|"
    r"scale|scaled|reward|advantage|response|responses|question|policy|sample|"
    r"sampled|computed|between|number|layers|tokens|total|average|only|note|"
    r"then|also|both|more|less|using|used|over|under|such|since|thus|here|"
    r"before|after|because|although|however|otherwise|additive|simplified|third|"
    r"commit|wait|follows|paired|normalized|calculating|calculated|normalizing)\b",
    re.IGNORECASE,
)
_SUP_TAG_RE = re.compile(r"<sup>([\s\S]*?)</sup>", re.IGNORECASE)
_SUB_TAG_RE = re.compile(r"<sub>([\s\S]*?)</sub>", re.IGNORECASE)

# Glued italic underscore patterns from PyMuPDF4LLM.
_DECIMAL_UNDER_DOT_RE = re.compile(r"(?<=\d)_\._(?=\d)")
_GLUED_PREFIX_KEYWORD_SUB_RE = re.compile(
    r"([0-9A-Za-z\u0370-\u03ff)])_([dn])_(model|ff|attn|layer|vocab|ctx|embd)(?=[A-Za-z0-9+\-*/()|\s.,;:]|$)"
)
_GLUED_KEYWORD_SUB_SUFFIX_RE = re.compile(
    r"(?<![A-Za-z0-9_])_([dn])_(model|ff|attn|layer|vocab|ctx|embd)(?=[0-9A-Za-z+\-*/()])"
)
_ITALIC_KEYWORD_SUB_RE = re.compile(r"(?<![A-Za-z0-9_])_([dn])_\s*(model|ff|attn|layer|vocab|ctx|embd|[kvqKVQ])\b")
_ITALIC_DFF_RE = re.compile(r"(?<![A-Za-z0-9_])_d(ff|model|attn|k|v|q)_(?![A-Za-z0-9_])")
_ITALIC_SET_BRACE_RE = re.compile(r"(?<![A-Za-z0-9_\u2211\u220f])_(\{[^{}\n_]{0,40}\}|[{}])_(?![A-Za-z0-9_])")
_ITALIC_SET_OUTER_RE = re.compile(r"(?<![A-Za-z0-9_\u2211\u220f])_(\{[^{}\n]{1,60}\})_(?![A-Za-z0-9_])")
_GLUED_BOTH_ITALIC_RE = re.compile(
    r"([A-Za-z0-9\u0370-\u03ff)\]*])_([A-Za-z0-9\u0370-\u03ff\u2192]{1,12})_(?=[A-Za-z0-9\u0370-\u03ff=+\-*/(])"
)
_GLUED_LEFT_ITALIC_RE = re.compile(
    r"(?<![A-Za-z0-9_])_([A-Za-z0-9\u0370-\u03ff\u2192\u2212+\-*/|]{1,18})_(?=[A-Za-z0-9\u0370-\u03ff=+\-*/(])"
)
_GLUED_RIGHT_ITALIC_RE = re.compile(
    r"(?<=[A-Za-z0-9)\]*])_([A-Za-z0-9\u0370-\u03ff\u2192\u2212+\-*/|]{1,18})_(?![A-Za-z0-9_])"
)
_STANDALONE_ITALIC_RE = re.compile(
    r"(?<![A-Za-z0-9_])_([^\s_\n][^_\n]{0,60}?[^\s_\n]|[^\s_\n])_(?![A-Za-z0-9_])"
)
_STRANDED_BOLD_ITALIC_RE = re.compile(r"\*\*_([^_\n*]{1,24})_\*\*")
_SINGLE_CHAR_BOLD_RE = re.compile(r"\*\*([A-Za-z0-9\u0370-\u03ff]{1,3})\*\*")

# Period missing space before sentence-starting Capitalized word (`2.Scaleγ` -> `2. Scale γ`).
_PERIOD_NO_SPACE_RE = re.compile(r"([0-9)\]])\.(?=[A-Z][a-z])")

# OCR spacing and math variable normalization.
_SPACE_IN_PARENS_BOTH_RE = re.compile(r"\(\s+([^()\n]{1,80}?)\s+\)")
_SPACE_AFTER_OPEN_PAREN_RE = re.compile(r"\(\s+([A-Za-z0-9\u0370-\u03ff|])")
_SPACE_BEFORE_CLOSE_PAREN_RE = re.compile(r"([A-Za-z0-9\u0370-\u03ff|])\s+\)")
_SPACE_BEFORE_COMMA_RE = re.compile(r"([A-Za-z0-9)\]*_\u0370-\u03ff])\s+,")
_SPACE_BEFORE_PERIOD_RE = re.compile(r"([A-Za-z0-9)\]\u0370-\u03ff])\s+\.(?=\s|$)")
_OCR_W_SUB_RE = re.compile(r"(\bxW|(?<![A-Za-z0-9_])W)\s+([12]|Q|K|V|O|q|k|v|o)\b")
_OCR_B_SUB_RE = re.compile(r"(?<![A-Za-z0-9_])b\s+([12])\b")
_OCR_D_SUB_RE = re.compile(r"(?<![A-Za-z0-9_])d\s+(model|ff|k|v|q|h|c)\b")
_OCR_PAREN_CLOSE_WORD_RE = re.compile(r"(\))([A-Z]_[12]|[A-Z]\b)")

# Math shorthand fixes (`σ2` -> `σ^2`, `(v_k - μ)2` -> `(v_k - μ)^2`, `dk` -> `d_k`, `vk` -> `v_k`).
_SIGMA_SQ_RE = re.compile(r"(?<![A-Za-z0-9_])\u03c3\s*_?\s*2(?![0-9.])")
_LATEX_SIGMA_SQ_RE = re.compile(r"\\sigma\s*_?\s*2(?![0-9.])")
_PAREN_SQ_RE = re.compile(r"(\([^()\n]{1,64}\))2(?![0-9A-Za-z])")
_SQ_EQ_SPACE_RE = re.compile(r"(\^[0-9A-Za-z]+)=")
_DK_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])dk(?![A-Za-z0-9_])")
_DV_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])dv(?![A-Za-z0-9_])")
_DQ_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])dq(?![A-Za-z0-9_])")
_VK_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])vk(?![A-Za-z0-9_])")
_SUM_FOLLOWED_VAR_RE = re.compile(
    r"((?:\u2211|\\sum)(?:_\{[^{}]+\}|\^\{[^{}]+\}|_[A-Za-z0-9]|\^[A-Za-z0-9]){1,2})(?=[A-Za-z\u0370-\u03ff(])"
)
_PAREN_FRAC_RE = re.compile(r"\(([^()\n]{1,40})\)/\(([^()\n]{1,40})\)")
_SQRT_GLUED_RE = re.compile(r"\u221a\s+([A-Za-z0-9_]+)")
_LATEX_SUB_SUP_BRACED_RE = re.compile(r"[_^](?:\{[^{}\n]{1,48}\}|\\[A-Za-z]+)")


def _normalize_math_unicode(text: str) -> str:
    """Map Mathematical Alphanumeric Unicode characters and micro sign to standard text/Greek."""
    out: List[str] = []
    for ch in text:
        cp = ord(ch)
        if 0x1D400 <= cp <= 0x1D7FF or cp in (0x210E, 0x2113, 0x00B5):
            out.append(unicodedata.normalize("NFKC", ch))
        elif cp == 0x2212:  # Unicode minus sign -> ASCII hyphen-minus
            out.append("-")
        else:
            out.append(ch)
    return "".join(out)


class _Protector:
    """Stash sensitive spans behind PUA sentinels so normalizers never corrupt them."""

    def __init__(self) -> None:
        self.items: List[str] = []

    def stash(self, value: str) -> str:
        idx = len(self.items)
        self.items.append(value)
        return f"{_SENTINEL_LEFT}{idx}{_SENTINEL_RIGHT}"

    def protect(self, text: str, pattern: re.Pattern[str], *, clean_inner: bool = False) -> str:
        def _repl(m: re.Match[str]) -> str:
            val = m.group(0)
            if clean_inner:
                val = _CTRL_CHARS_RE.sub("", val).replace("\ufffd", "")
            return self.stash(val)

        return pattern.sub(_repl, text)

    def restore(self, text: str) -> str:
        if not self.items:
            return text
        pattern = re.compile(re.escape(_SENTINEL_LEFT) + r"(\d+)" + re.escape(_SENTINEL_RIGHT))
        for _ in range(4):
            if _SENTINEL_LEFT not in text:
                break
            text = pattern.sub(lambda m: self.items[int(m.group(1))], text)
        return text


def _strip_unmatched_bold_per_line(text: str) -> str:
    """Remove unmatched/stray `**` markers while keeping balanced bold spans intact."""
    lines: List[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"\*\*\s*\*\*", "", line)
        cleaned = re.sub(r"^\s*\*\*\s*(?=[:;.,])", "", cleaned)
        if cleaned.count("**") % 2 == 1:
            cleaned = cleaned.replace("**", "")
        lines.append(cleaned)
    return "\n".join(lines)


def _replace_sup_tags(text: str) -> str:
    """Convert short mathematical `<sup>` tags to `^...` and unwrap prose-swallowing `<sup>`."""

    def _repl(m: re.Match[str]) -> str:
        inner = m.group(1)
        start = m.start()
        end = m.end()
        before = text[max(0, start - 6) : start]
        after = text[end : min(len(text), end + 6)]

        compact = re.sub(r"[*_\s]+", "", inner)
        if not compact:
            return " " if before and after and before[-1].isalnum() and after[0].isalnum() else ""

        has_valid_base = bool(
            before
            and not before[-1].isspace()
            and not before.endswith(("=1", "=0", "</sup>", "\u2211", "\u220f", "(", "[", ",", ";", ":", "=", "+", "-", "*", "/"))
            and (before[-1].isalnum() or before[-1] in {")", "]", "_", "*"})
        )
        if before.endswith((".", ",")) and compact.isdigit():
            has_valid_base = True

        is_prose = (
            not has_valid_base
            or len(compact) > 10
            or bool(_PROSE_WORDS_RE.search(inner))
            or ("=" in inner)
            or (". " in inner)
            or bool(re.search(r"[A-Za-z]\.[A-Za-z]", inner))
            or bool(re.search(r"[<>]", inner))
            or compact in {"(", ")", "[", "]", ",", ".", ";", ":", "\u00b7", "\u00d7", "+", "-", "\u2212"}
            or ("_" in inner and len(compact) > 4 and not re.fullmatch(r"_?[A-Za-z]{1,4}_?(?:model|ff|attn|DKV|KV)?", compact))
        )
        if is_prose:
            unwrapped = inner.strip()
            unwrapped = re.sub(r"([.,;:])(?=[A-Za-z])", r"\1 ", unwrapped)
            pad_l = " " if before and before[-1] not in " \n\t([{" and not before.endswith("<sup>") else ""
            pad_r = " " if after and after[0] not in " \n\t.,;:!?)]}<" else ""
            return f"{pad_l}{unwrapped}{pad_r}"

        clean_sup = inner.replace("_", "").replace("**", "").strip()
        if not clean_sup:
            return ""
        if clean_sup == "\u221a":
            return "\u221a"
        if len(clean_sup) == 1 and clean_sup.isalnum():
            out_sup = f"^{clean_sup}"
        else:
            out_sup = f"^{{{clean_sup}}}"
        pad_r = " " if after and after[0].isalpha() and not clean_sup.endswith("(") else ""
        return f"{out_sup}{pad_r}"

    return _SUP_TAG_RE.sub(_repl, text)


def _unwrap_glued_underscores(text: str, protector: _Protector) -> str:
    """Unwrap PyMuPDF italic `_..._` markers while strictly protecting real `snake_case` IDs."""
    s = _DECIMAL_UNDER_DOT_RE.sub(".", text)
    # Fix `4_d_model`, `2_d_model_n_layer`, `2_d_attn` before protecting valid snake_case.
    for _ in range(3):
        prev = s
        s = _GLUED_PREFIX_KEYWORD_SUB_RE.sub(r"\1 \2_\3", s)
        s = _GLUED_KEYWORD_SUB_SUFFIX_RE.sub(r"\1_\2 ", s)
        if s == prev:
            break
    s = _ITALIC_KEYWORD_SUB_RE.sub(r"\1_\2", s)
    s = _ITALIC_DFF_RE.sub(r"d_\1", s)
    s = _STRANDED_BOLD_ITALIC_RE.sub(r"\1", s)
    s = _SINGLE_CHAR_BOLD_RE.sub(r"\1", s)
    s = _ITALIC_SET_BRACE_RE.sub(r"\1", s)

    # Protect legitimate snake_case IDs (`vaswani_2017_transformer`, `d_model`, `core_block`, etc.)
    # and braced LaTeX subscripts/superscripts (`_{k=1}`, `^{d_k}`, `^{DKV}`) FIRST.
    s = protector.protect(s, _VALID_SNAKE_CASE_RE)
    s = protector.protect(s, _LATEX_SUB_SUP_BRACED_RE)

    # Unwrap glued italic spans like `_vk_and_σ_2=`, `of_m_HC`, `_G_becomes`, `2_d_`.
    for _ in range(4):
        prev = s
        s = _GLUED_LEFT_ITALIC_RE.sub(r"\1 ", s)
        s = _GLUED_BOTH_ITALIC_RE.sub(r"\1 \2 ", s)
        s = _GLUED_RIGHT_ITALIC_RE.sub(r" \1", s)
        if s == prev:
            break

    # Protect clean single-underscore math variables (`d_k`, `v_k`, `xW_1`, `b_1`) BEFORE
    # unwrapping standalone multi-word italic spans like `_x, y_` or `_N_`.
    s = protector.protect(
        s,
        re.compile(r"(?<![A-Za-z0-9_])[A-Za-z\u0370-\u03ff]{1,3}_[A-Za-z0-9\u0370-\u03ff]{1,3}(?![A-Za-z0-9_])"),
    )

    # Unwrap remaining `_italic_` spans (`_a_` -> `a`, `_N_` -> `N`, `_,_` -> `,`, `_x, y_` -> `x, y`).
    for _ in range(3):
        prev = s
        s = _STANDALONE_ITALIC_RE.sub(r"\1", s)
        s = _ITALIC_SET_OUTER_RE.sub(r"\1", s)
        if s == prev:
            break

    # Strip orphan single-sided underscores left by line/chunk splits (`_x,` or `N_`),
    # never touching protected sentinels or `{...}` braces.
    s = re.sub(r"(?<![A-Za-z0-9_{])_(?=[A-Za-z0-9\u0370-\u03ff(<\[{\u2308.,;:])", "", s)
    s = re.sub(r"(?<=[A-Za-z0-9\u0370-\u03ff)>.,;:\]\u2309}])_(?![A-Za-z0-9_{])", "", s)
    return s


def _repair_pdf_fractions_and_fffd(text: str) -> str:
    """Repair PDF fraction `<sup><u>...</u></sup>` / `<u>...</u>` constructs and `\ufffd`."""
    s = _TILDE_SQRT_RE.sub("\u221a", text)
    s = _TILDE_WRAP_RE.sub(r"\1", s)
    s = s.replace("~~", "")

    s = _FRAC_NUM_SPLIT_RE.sub(
        lambda m: f"({m.group(1).replace('_', '').strip()} - {m.group(3).replace('_', '').strip()})/{m.group(2).replace('_', '')}",
        s,
    )
    s = _FRAC_JACOBIAN_RE.sub(lambda m: f"\u2202{m.group(1)}({m.group(3)})/\u2202{m.group(2)}", s)
    s = _FRAC_SQRT_K_UNDER_1_RE.sub(lambda m: f"\u221a(1/{m.group(1)}) \u2211", s)

    def _den_sup_u(m: re.Match[str]) -> str:
        den = m.group(1)
        num = m.group(2).replace("_", "").strip()
        sub = m.group(3)
        if sub:
            den = f"{den}_{sub}"
        return f"{num}/{den}"

    s = _FRAC_DEN_SUP_U_RE.sub(_den_sup_u, s)

    def _eq_sup_u(m: re.Match[str]) -> str:
        lead = m.group(1)
        num = m.group(2).replace("_", "").strip()
        den = m.group(3).replace("_", "").strip()
        prime = m.group(4) or ""
        return f"{lead}{num}/{den}{prime}"

    s = _FRAC_EQ_SUP_U_RE.sub(_eq_sup_u, s)
    s = _FRAC_SQRT_UNDER_1_RE.sub(lambda m: f"1/\u221a{m.group(1).replace('_', '')}", s)

    s = _UNDERLINE_TAG_RE.sub(r"\1", s)
    s = re.sub(r"</?u>", "", s, flags=re.IGNORECASE)

    s = _MULTI_FFFD_RE.sub("", s)
    s = _FFFD_BEFORE_SUM_RE.sub("\u2211", s)
    s = s.replace("\ufffd", "")
    s = _SUM_SUP_LIMIT_RE.sub(lambda m: f"\u2211_{{{m.group(2)}=1}}^{{{m.group(1)}}}", s)
    s = _SUM_COMBINED_LIMITS_RE.sub(lambda m: f"\u2211_{{{m.group(2)}=1}}^{{{m.group(1)}}}", s)
    s = _SUM_LOWER_ONLY_RE.sub(lambda m: f"\u2211_{{{m.group(1)}={m.group(2)}}}", s)
    return s


def _clean_ocr_and_math_spacing(text: str, *, latex_frac: bool) -> str:
    """Fix PDF/OCR spacing around parens, periods, subscripts, and broken inline equations."""
    s = _PERIOD_NO_SPACE_RE.sub(r"\1. ", text)
    s = _SPACE_IN_PARENS_BOTH_RE.sub(r"(\1)", s)
    s = _SPACE_AFTER_OPEN_PAREN_RE.sub(r"(\1", s)
    s = _SPACE_BEFORE_CLOSE_PAREN_RE.sub(r"\1)", s)
    s = _SPACE_BEFORE_COMMA_RE.sub(r"\1,", s)
    s = _SPACE_BEFORE_PERIOD_RE.sub(r"\1.", s)

    s = _OCR_W_SUB_RE.sub(r"\1_\2", s)
    s = _OCR_B_SUB_RE.sub(r"b_\1", s)
    s = _OCR_D_SUB_RE.sub(r"d_\1", s)
    s = _OCR_PAREN_CLOSE_WORD_RE.sub(r"\1\2", s)

    s = _SIGMA_SQ_RE.sub("\u03c3^2", s)
    s = _LATEX_SIGMA_SQ_RE.sub(r"\\sigma^2" if latex_frac else "\u03c3^2", s)
    s = _PAREN_SQ_RE.sub(r"\1^2", s)
    s = _SQ_EQ_SPACE_RE.sub(r"\1 =", s)
    s = _DK_TOKEN_RE.sub("d_k", s)
    s = _DV_TOKEN_RE.sub("d_v", s)
    s = _DQ_TOKEN_RE.sub("d_q", s)
    s = _VK_TOKEN_RE.sub("v_k", s)
    s = _SQRT_GLUED_RE.sub("\u221a\\1", s)
    s = re.sub(r"(?<=[A-Za-z0-9}\)])(?=\u2211|\\sum)", " ", s)
    s = re.sub(r"(?<=\s)-(?=[\u03bc\u03c3\u03b1\u03b2\u03b3\u03b4\u03b5\u03b8\u03bb\u03c0])", "- ", s)
    s = _SUM_FOLLOWED_VAR_RE.sub(r"\1 ", s)

    if latex_frac:
        s = _PAREN_FRAC_RE.sub(r"\\frac{\1}{\2}", s)
    else:
        s = _PAREN_FRAC_RE.sub(r"\1/\2", s)
    return s


def clean_section_path(path: str) -> str:
    """Strip PyMuPDF4LLM bold/italic/HTML artifacts from section headings while keeping snake_case intact."""
    s = str(path or "")
    if not s:
        return ""
    protector = _Protector()
    s = protector.protect(s, _VALID_SNAKE_CASE_RE)
    s = _HTML_COMMENT_RE.sub("", s)
    s = _BR_TAG_RE.sub(" ", s)
    s = _CTRL_CHARS_RE.sub("", s)
    s = _INVISIBLES_RE.sub("", s)
    s = _normalize_math_unicode(s)
    s = _UNDERLINE_TAG_RE.sub(r"\1", s)
    s = re.sub(r"</?(?:u|sup|sub)>", "", s, flags=re.IGNORECASE)
    s = s.replace("\ufffd", "")
    s = _SEC_BOLD_RE.sub(r"\1", s)
    s = s.replace("**", "")
    s = _SEC_EMPH_UNDER_RE.sub(r"\1", s)
    s = _SEC_EMPH_STAR_RE.sub(r"\1", s)
    s = _SEC_STRAY_STAR_RE.sub("", s)
    s = protector.restore(s)
    s = re.sub(r"(?:\s*>\s*)+$", "", s)
    return re.sub(r"[ \t]{2,}", " ", s).strip()


def _clean_core(text: str, *, latex_frac: bool, strip_all_stray_bold: bool) -> str:
    s = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not s:
        return ""

    s = _EMAIL_ANGLE_RE.sub(lambda m: f"<{m.group(1)}>", s)

    protector = _Protector()
    s = protector.protect(s, _CODE_BLOCK_RE, clean_inner=True)
    s = protector.protect(s, _URL_RE, clean_inner=True)
    s = protector.protect(s, CITATION_RE)
    s = protector.protect(s, _TOKEN_TAG_RE)
    s = protector.protect(s, _EMAIL_ANGLE_RE)
    s = protector.protect(s, _EINSUM_STR_RE)

    s = _HTML_COMMENT_RE.sub("", s)
    s = _BR_TAG_RE.sub("\n", s)
    s = _CTRL_CHARS_RE.sub("", s)
    s = _INVISIBLES_RE.sub("", s)
    s = _normalize_math_unicode(s)

    s = _repair_pdf_fractions_and_fffd(s)
    s = _replace_sup_tags(s)
    s = _SUB_TAG_RE.sub(lambda m: f"_{{{m.group(1).replace('_', '').strip()}}}" if m.group(1).strip() else "", s)
    s = re.sub(r"</?(?:sup|sub|u)>", "", s, flags=re.IGNORECASE)

    s = _unwrap_glued_underscores(s, protector)
    s = _strip_unmatched_bold_per_line(s)
    if strip_all_stray_bold:
        s = re.sub(r"\*\*\s*$", "", s, flags=re.MULTILINE)

    s = protector.restore(s)
    protector2 = _Protector()
    s = protector2.protect(s, _CODE_BLOCK_RE)
    s = protector2.protect(s, _URL_RE)
    s = protector2.protect(s, CITATION_RE)
    s = protector2.protect(s, _TOKEN_TAG_RE)
    s = protector2.protect(s, _EMAIL_ANGLE_RE)
    s = protector2.protect(s, _EINSUM_STR_RE)
    s = protector2.protect(s, _VALID_SNAKE_CASE_RE)

    s = _clean_ocr_and_math_spacing(s, latex_frac=latex_frac)
    s = protector2.restore(s)

    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def clean_passage_text(text: str) -> str:
    """Clean extracted PDF passage text without altering mathematical meaning, citations, or snake_case."""
    return _clean_core(text, latex_frac=False, strip_all_stray_bold=False)


def clean_generated_answer(answer: str) -> str:
    """Safely normalize LLM answers without breaking `[chunk_id]` citations, `W^{DKV}`, or `snake_case`."""
    return _clean_core(answer, latex_frac=True, strip_all_stray_bold=True)
