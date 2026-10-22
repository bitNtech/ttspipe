"""Edge cases for the incremental text segmenter in `app.segmenter`."""

from __future__ import annotations

import pytest

from app.segmenter import Segmenter


def _drive(segmenter: Segmenter, fragments: list[str], finish: str = "close") -> list[str]:
    out: list[str] = []
    for frag in fragments:
        out += segmenter.push(frag)
    out += getattr(segmenter, finish)()
    return out


def test_flushes_on_sentence_end_once_min_reached():
    s = Segmenter(min_segment_chars=10, max_segment_chars=200)
    out = s.push("Hi. This is long enough now. ")
    # "Hi." alone (3 chars) is below min_segment_chars, so it should not be
    # cut off on its own -- it waits for the buffer to reach min length.
    assert out == ["Hi. This is long enough now."]
    assert s.close() == []


def test_short_sentence_end_is_held_until_min_reached():
    s = Segmenter(min_segment_chars=50, max_segment_chars=200)
    out = s.push("Hi. Still short. ")
    assert out == []  # neither sentence end alone clears 50 chars
    out += s.push("Now the buffer is finally long enough to flush.")
    assert out == ["Hi. Still short. Now the buffer is finally long enough to flush."]


def test_multiple_sentences_in_one_fragment():
    s = Segmenter(min_segment_chars=5, max_segment_chars=200)
    out = s.push("First sentence. Second sentence. Third one.")
    assert out == ["First sentence.", "Second sentence.", "Third one."]


def test_hard_cut_at_max_chars_on_word_boundary():
    s = Segmenter(min_segment_chars=1, max_segment_chars=20)
    text = "This sentence has no punctuation at all so it must be hard cut"
    out = s.push(text)
    out += s.close()
    assert out  # at least one hard cut happened
    for seg in out:
        assert len(seg) <= 20
        assert seg == seg.strip()
    # every cut lands on a word boundary: rejoining with single spaces
    # reproduces the original text exactly (no word was torn in half)
    assert " ".join(out) == text


def test_hard_cut_may_split_a_single_giant_word_as_a_last_resort():
    s = Segmenter(min_segment_chars=1, max_segment_chars=15)
    text = "supercalifragilisticexpialidocious is very long"
    out = s.push(text)
    out += s.close()
    for seg in out:
        assert len(seg) <= 15
    # no characters lost or reordered, even though the giant word (with no
    # internal boundary) necessarily gets cut mid-word
    assert "".join(out).replace(" ", "") == text.replace(" ", "")


def test_cjk_text_with_no_spaces():
    s = Segmenter(min_segment_chars=5, max_segment_chars=12)
    out = s.push("今日は良い天気ですね。散歩に行きましょう")
    out += s.close()
    assert out[0] == "今日は良い天気ですね。"
    assert "".join(out) == "今日は良い天気ですね。散歩に行きましょう"


def test_abbreviation_dr_not_split():
    s = Segmenter(min_segment_chars=1, max_segment_chars=200)
    out = s.push("Dr. Smith arrived. He was early.")
    assert out == ["Dr. Smith arrived.", "He was early."]


def test_decimal_not_split():
    s = Segmenter(min_segment_chars=1, max_segment_chars=200)
    out = s.push("The price is 3.5 dollars today.")
    out += s.close()
    assert out == ["The price is 3.5 dollars today."]


def test_decimal_split_across_fragments_not_split():
    s = Segmenter(min_segment_chars=1, max_segment_chars=200)
    out = s.push("The price is 3.")
    assert out == []  # ambiguous: could be a decimal, held back
    out += s.push("5 dollars exactly.")
    out += s.close()
    assert out == ["The price is 3.5 dollars exactly."]


def test_tag_never_split_across_segments():
    s = Segmenter(min_segment_chars=1, max_segment_chars=15)
    out = s.push("I feel great [laugh] today my friends")
    out += s.close()
    joined = "".join(out)
    assert "[laugh]" in "".join(seg for seg in out if "[laugh]" in seg)
    # the tag must appear whole inside exactly one segment
    tag_segments = [seg for seg in out if "[laugh]" in seg]
    assert len(tag_segments) == 1
    assert joined.replace(" ", "") == "Ifeelgreat[laugh]todaymyfriends"


def test_tag_split_across_fragments_still_not_split():
    s = Segmenter(min_segment_chars=1, max_segment_chars=200)
    out = s.push("Say it now [wh")
    assert out == []  # tag incomplete, must not cut inside/around it yet
    out += s.push("ispers]very quietly.")
    out += s.close()
    assert out == ["Say it now [whispers]very quietly."]


def test_newline_triggers_flush_once_min_reached():
    s = Segmenter(min_segment_chars=5, max_segment_chars=200)
    out = s.push("Line one is long enough\nLine two")
    assert out == ["Line one is long enough"]
    out += s.close()
    assert out == ["Line one is long enough", "Line two"]


def test_flush_drains_remainder_as_one_segment():
    s = Segmenter(min_segment_chars=100, max_segment_chars=200)
    s.push("Too short to auto-flush")
    out = s.flush()
    assert out == ["Too short to auto-flush"]
    assert s.pending == ""


def test_flush_on_empty_buffer_yields_nothing():
    s = Segmenter(min_segment_chars=5, max_segment_chars=200)
    assert s.flush() == []
    assert s.close() == []


def test_close_is_idempotent_and_final():
    s = Segmenter(min_segment_chars=5, max_segment_chars=200)
    s.push("some text")
    first = s.close()
    second = s.close()
    assert first == ["some text"]
    assert second == []


def test_semicolon_and_colon_are_sentence_final():
    s = Segmenter(min_segment_chars=1, max_segment_chars=200)
    out = s.push("First clause; second clause: third clause.")
    assert out == ["First clause;", "second clause:", "third clause."]


def test_colon_in_time_like_pattern_not_split():
    s = Segmenter(min_segment_chars=1, max_segment_chars=200)
    out = s.push("We meet at 12:30 sharp.")
    out += s.close()
    assert out == ["We meet at 12:30 sharp."]


def test_constructor_rejects_nonpositive_bounds():
    with pytest.raises(ValueError):
        Segmenter(min_segment_chars=0, max_segment_chars=0)


# --------------------------------------------------------------------------
# Unclosed brackets must not hold the whole turn hostage
# --------------------------------------------------------------------------


def test_an_unclosed_bracket_does_not_stall_the_segmenter():
    """Holding the cut back at `[` is right for `[laugh]`, which closes in a
    few characters. For a bracket that never closes - "the array [0", a tag an
    upstream LLM abandoned - it used to freeze the scan limit forever, so no
    later boundary could be found and nothing was emitted until `flush`.
    Measured before the bound: 950 characters buffered, zero segments, i.e. the
    voice goes silent for the rest of the turn."""
    segmenter = Segmenter(24, 220)
    segments = segmenter.push("Sure, let me check the array [0 and keep talking. ")
    for _ in range(6):
        segments += segmenter.push("This is another full sentence that should be cut by now. ")

    assert segments, "an unclosed bracket stalled the segmenter"
    assert len(segmenter.pending) < 220


def test_a_real_tag_is_still_never_split():
    segmenter = Segmenter(24, 220)
    segments = segmenter.push("That is hilarious [laugh] and then we continue onward here. ")
    assert segments == ["That is hilarious [laugh] and then we continue onward here."]


def test_a_bracket_is_held_while_it_still_looks_like_a_tag():
    """Short unclosed brackets keep their protection - the cut is pulled back
    so a tag in flight is never sliced in half."""
    segmenter = Segmenter(4, 40)
    segments = segmenter.push("Hello there friend. [whispe")
    assert segments == ["Hello there friend."]
    assert segmenter.pending.strip() == "[whispe"


def test_zero_min_segment_chars_is_rejected():
    """A zero minimum lets `_find_cut` return 0, and `push` then loops forever
    on a buffer it never shortens - synchronously, on the event loop."""
    with pytest.raises(ValueError, match="at least 1"):
        Segmenter(0, 220)


# --------------------------------------------------------------------------
# Punctuation must stay with the sentence it punctuates
# --------------------------------------------------------------------------


def test_an_ellipsis_is_not_split_across_segments():
    """Cutting after the first dot leaves ".." opening the next segment - text
    the model then has to find a pronunciation for."""
    segmenter = Segmenter(24, 220)
    segments = segmenter.push(
        "Well that was a long thought... and then we carried on talking for a while. "
    )
    assert segments[0].endswith("...")
    assert not segments[1].startswith(".")


def test_a_closing_quote_stays_with_its_sentence():
    segmenter = Segmenter(24, 220)
    segments = segmenter.push(
        'He said "Hello there friend." Then he walked away without saying more. '
    )
    assert segments[0].endswith('."')
    assert not segments[1].startswith('"')


def test_interrobang_is_kept_whole():
    segmenter = Segmenter(4, 220)
    segments = segmenter.push("Are you serious?! I really cannot believe that. ")
    assert segments[0] == "Are you serious?!"
    assert not segments[1].startswith("!")


def test_an_ordinary_full_stop_still_cuts_where_it_did():
    segmenter = Segmenter(4, 220)
    assert segmenter.push("Hello there. Second one. ") == ["Hello there.", "Second one."]


def test_first_segment_is_cut_early_then_the_minimum_applies():
    """A reply's opener is spoken at once; later short sentences still merge."""
    s = Segmenter(min_segment_chars=24, max_segment_chars=200, first_min_chars=2)
    assert s.push("Sure. ") == ["Sure."]
    assert s.push("Okay. ") == []  # below 24 now, so it waits for company
    assert s.push("Your order shipped this morning. ") == [
        "Okay. Your order shipped this morning."
    ]


def test_restart_reapplies_the_first_segment_rule():
    s = Segmenter(min_segment_chars=24, max_segment_chars=200, first_min_chars=2)
    assert s.push("Sure. ") == ["Sure."]
    s.push("half a thought")
    s.restart()  # barge-in: the buffered text is dropped
    assert s.pending == ""
    assert s.push("Right. ") == ["Right."]


def test_first_min_defaults_to_the_regular_minimum():
    s = Segmenter(min_segment_chars=24, max_segment_chars=200)
    assert s.push("Sure. ") == []
