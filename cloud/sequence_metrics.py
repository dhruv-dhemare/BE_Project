from typing import List, Tuple


def edit_distance(left, right) -> int:
    table = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        previous = table[0]
        table[0] = i
        for j, right_item in enumerate(right, start=1):
            saved = table[j]
            table[j] = min(
                table[j] + 1,
                table[j - 1] + 1,
                previous + (left_item != right_item),
            )
            previous = saved
    return table[-1]


def cer(reference: str, hypothesis: str) -> float:
    return edit_distance(list(reference), list(hypothesis)) / max(1, len(reference))


def wer(reference: str, hypothesis: str) -> float:
    return edit_distance(reference.split(), hypothesis.split()) / max(1, len(reference.split()))


def batch_cer_wer(references: List[str], hypotheses: List[str]) -> Tuple[float, float]:
    char_errors = sum(edit_distance(list(ref), list(hyp)) for ref, hyp in zip(references, hypotheses))
    char_total = sum(max(1, len(ref)) for ref in references)
    word_errors = sum(edit_distance(ref.split(), hyp.split()) for ref, hyp in zip(references, hypotheses))
    word_total = sum(max(1, len(ref.split())) for ref in references)
    return char_errors / char_total, word_errors / word_total
