from branchmi_pilot.answers import NO_ANSWER, canonicalize_answer, extract_final_answer


def test_extract_nested_boxed_answer():
    text = r"Reasoning. Therefore the final result is \boxed{\frac{1}{2}}."
    assert extract_final_answer(text) == r"\frac{1}{2}"


def test_answer_tag_has_priority():
    text = r"Earlier \boxed{3}. <answer>4</answer>"
    assert extract_final_answer(text) == "4"


def test_empty_answer():
    assert extract_final_answer("  ") == NO_ANSWER


def test_canonicalization_removes_cosmetic_latex():
    assert canonicalize_answer(r"$\left 2 \, + 3\right$") == "2+3"

