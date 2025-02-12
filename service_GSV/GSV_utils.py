
import re
from GPT_SoVITS.text.cleaner import clean_text
from GPT_SoVITS.text import cleaned_text_to_sequence

def cut5(inp):
    inp = inp.strip("\n")
    punds = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)


def process_text(texts):
    _text = []
    if all(text in [None, " ", "\n", ""] for text in texts):
        raise ValueError("请输入有效文本")
    for text in texts:
        if text in [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


def get_first(text, splits=None):
    if splits is None:
        splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def replace_consecutive_punctuation(text):
    punctuation = set(['!', '?', '…', ',', '.', '-', " "])
    punctuations = ''.join(re.escape(p) for p in punctuation)
    pattern = f'([{punctuations}])([{punctuations}])+'
    result = re.sub(pattern, r'\1', text)
    return result


def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text
