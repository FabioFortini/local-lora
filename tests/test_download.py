import random

import pytest
from datasets import ClassLabel, Dataset, Features, Value

from download import assign_splits, label_order, pick_label_column, pick_text_column, to_rows

BANKING = Features({"text": Value("string"), "label": Value("int64"), "label_text": Value("string")})


def test_label_column_prefers_classlabel():
    f = Features({"sentence": Value("string"), "label": ClassLabel(names=["neg", "pos"]), "idx": Value("int32")})
    assert pick_label_column(f) == "label"


def test_label_column_skips_int_without_names():
    # mteb/banking77: "label" is an unnamed int, "label_text" holds the names
    assert pick_label_column(BANKING) == "label_text"


def test_label_column_override_must_be_usable():
    with pytest.raises(SystemExit, match="ClassLabel"):
        pick_label_column(BANKING, "label")
    with pytest.raises(SystemExit, match="assente"):
        pick_label_column(BANKING, "nope")


def test_label_column_none_found():
    with pytest.raises(SystemExit, match="--label-column"):
        pick_label_column(Features({"text": Value("string"), "score": Value("float32")}))


def test_text_column():
    assert pick_text_column(BANKING, "label_text") == "text"
    f = Features({"premise": Value("string"), "label": ClassLabel(names=["a", "b"])})
    assert pick_text_column(f, "label") == "premise"
    f = Features({"premise": Value("string"), "hypothesis": Value("string"), "label": ClassLabel(names=["a", "b"])})
    with pytest.raises(SystemExit, match="--text-column"):
        pick_text_column(f, "label")
    assert pick_text_column(f, "label", "hypothesis") == "hypothesis"


def test_label_order_classlabel():
    f = Features({"text": Value("string"), "label": ClassLabel(names=["z", "a"])})
    assert label_order(f, "label", {}) == ["z", "a"]


def test_label_order_id_text_pair():
    split = Dataset.from_dict({"text": ["x", "y", "w"], "label": [1, 0, 1], "label_text": ["b", "z", "b"]},
                              features=BANKING)
    assert label_order(BANKING, "label_text", {"train": split}) == ["z", "b"]


def test_label_order_falls_back_when_not_one_to_one():
    split = Dataset.from_dict({"text": ["x", "y"], "label": [0, 0], "label_text": ["a", "b"]}, features=BANKING)
    assert label_order(BANKING, "label_text", {"train": split}) is None
    f = Features({"text": Value("string"), "category": Value("string")})
    assert label_order(f, "category", {}) is None


def test_to_rows_maps_classlabel_and_drops_unlabeled():
    f = Features({"text": Value("string"), "label": ClassLabel(names=["neg", "pos"])})
    split = Dataset.from_dict({"text": ["good", "bad", "unlabeled", ""], "label": [1, 0, -1, 1]}, features=f)
    rows, dropped = to_rows(split, "text", "label")
    assert rows == [{"text": "good", "label": "pos"}, {"text": "bad", "label": "neg"}]
    assert dropped == 2


def test_assign_splits_matches_original_banking77_algorithm():
    train = [{"i": i} for i in range(100)]
    test = [{"t": 1}]
    new_train, valid, new_test, origin = assign_splits(train, None, test, 0.1, 0.1, 42)
    expected = list(train)
    random.Random(42).shuffle(expected)
    assert valid == expected[:10] and new_train == expected[10:] and new_test == test
    assert origin == {"train": "train", "valid": "train", "test": "test"}


def test_assign_splits_carves_test_then_valid():
    train, valid, test, origin = assign_splits([{"i": i} for i in range(100)], None, None, 0.1, 0.2, 0)
    assert (len(train), len(valid), len(test)) == (72, 8, 20)
    assert origin["test"] == origin["valid"] == "train"
    assert not {r["i"] for r in test} & {r["i"] for r in train + valid}


def test_assign_splits_keeps_provided_splits():
    train, valid, test, _ = assign_splits([{"i": 1}, {"i": 2}], [{"v": 1}], [{"t": 1}], 0.1, 0.1, 0)
    assert sorted(r["i"] for r in train) == [1, 2] and valid == [{"v": 1}] and test == [{"t": 1}]


def test_assign_splits_too_small():
    with pytest.raises(SystemExit, match="vuoti"):
        assign_splits([{"i": 1}, {"i": 2}], None, None, 0.1, 0.1, 0)
