from toy.reward import parse_answer, reward

def test_correct_with_working():
    c = "17*23 = 17*20 + 17*3 = 340 + 51 = 391. <answer>391</answer>"
    assert parse_answer(c) == 391
    assert reward(c, 391) == 1.0

def test_negative():
    c = "<answer>-12</answer>"
    assert parse_answer(c) == -12
    assert reward(c, -12) == 1.0

def test_spaces_and_newlines_inside_tag():
    c = "<answer>\n 391 \n</answer>"
    assert parse_answer(c) == 391
    assert reward(c, 391) == 1.0

def test_several_tags_uses_last():      #last tag counts
    c = "<answer>5</answer> wait, let me redo it. <answer>391</answer>"
    assert parse_answer(c) == 391
    assert reward(c, 391) == 1.0

def test_wrong_number():
    c = "<answer>390</answer>"
    assert parse_answer(c) == 390
    assert reward(c, 391) == 0.0

def test_no_tag():                       #what the smoke test produced
    c = "The product of 17 and 23 is 391."
    assert parse_answer(c) is None
    assert reward(c, 391) == 0.0

def test_unclosed_tag():
    c = "<answer>391"
    assert parse_answer(c) is None
    assert reward(c, 391) == 0.0

def test_two_numbers_in_tag():           #hedging
    c = "<answer> 390 391</answer>"
    assert parse_answer(c) is None
    assert reward(c, 391) == 0.0

def test_words_in_tag():                #hedging with words
    c = "<answer>391 or 390</answer>"
    assert parse_answer(c) is None
    assert reward(c, 391) == 0.0

def test_comma_rejected():             #commas not allowed
    c = "<answer>1,234</answer>"
    assert parse_answer(c) is None
    assert reward(c, 1234) == 0.0

def test_space_inside_number():
    c = "<answer>3 91</answer>"
    assert parse_answer(c) is None
    assert reward(c, 391) == 0.0

def test_empty_tag():
    c = "<answer></answer>"
    assert parse_answer(c) is None
    assert reward(c, 391) == 0.0

def test_number_only_outside_tag():
    c = "391 <answer>forgot</answer>"
    assert parse_answer(c) is None
    assert reward(c, 391) == 0.0

def test_last_tag_wrong():         #last tag counts, even if an earlier one was right
    c = "<answer>391</answer> no, actually <answer>390</answer>"
    assert parse_answer(c) == 390
    assert reward(c, 391) == 0.0