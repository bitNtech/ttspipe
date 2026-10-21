"""Coverage for the wire protocol contract in `app.protocol`.

`app/protocol.py` is a fixed contract owned by another agent -- these tests
exercise it as a black box: the happy path for every client event, every way
a frame can be malformed, and that the server-side encoders round-trip
through msgpack cleanly.
"""

from __future__ import annotations

import base64
import json

import ormsgpack
import pytest
from pydantic import ValidationError

from app.protocol import (
    CancelEvent,
    TTSParams,
    FlushEvent,
    PingEvent,
    ProtocolError,
    StartEvent,
    StopEvent,
    TextEvent,
    audio_frame,
    decode_client_frame,
    finish_frame,
    log_frame,
    pong_frame,
    started_frame,
)


def _msgpack(payload: dict) -> bytes:
    return ormsgpack.packb(payload)


# -- happy path, one per client event --


def test_decode_start_msgpack_defaults():
    event = decode_client_frame(_msgpack({"event": "start"}))
    assert isinstance(event, StartEvent)
    assert event.request.text == ""
    assert event.request.format == "wav"
    assert event.request.seed is None
    assert event.debug is False


def test_decode_start_with_full_request():
    payload = {
        "event": "start",
        "request": {
            "text": "hello",
            "format": "pcm",
            "seed": 42,
            "temperature": 0.5,
            "top_p": 0.9,
        },
        "debug": True,
    }
    event = decode_client_frame(_msgpack(payload))
    assert isinstance(event, StartEvent)
    assert event.request.format == "pcm"
    assert event.request.seed == 42
    assert event.request.temperature == 0.5
    assert event.debug is True


def test_decode_start_with_reference_audio_base64():
    raw_audio = b"\x00\x01\x02\x03fake-wav-bytes"
    payload = {
        "event": "start",
        "request": {
            "references": [{"audio": base64.b64encode(raw_audio).decode("ascii"), "text": "hi"}]
        },
    }
    event = decode_client_frame(_msgpack(payload))
    assert len(event.request.references) == 1
    assert event.request.references[0].audio == raw_audio
    assert event.request.references[0].text == "hi"


def test_decode_text_event():
    event = decode_client_frame(_msgpack({"event": "text", "text": "a fragment "}))
    assert isinstance(event, TextEvent)
    assert event.text == "a fragment "


def test_decode_flush_stop_cancel_ping():
    assert isinstance(decode_client_frame(_msgpack({"event": "flush"})), FlushEvent)
    assert isinstance(decode_client_frame(_msgpack({"event": "stop"})), StopEvent)
    assert isinstance(decode_client_frame(_msgpack({"event": "cancel"})), CancelEvent)
    assert isinstance(decode_client_frame(_msgpack({"event": "ping"})), PingEvent)


def test_decode_accepts_json_text_frame():
    event = decode_client_frame(json.dumps({"event": "text", "text": "from json"}))
    assert isinstance(event, TextEvent)
    assert event.text == "from json"


def test_decode_json_bytes_over_binary_frame_is_not_auto_detected():
    # A JSON object's leading '{' (0x7B) is itself a valid single-byte msgpack
    # positive-fixint, so raw JSON bytes over a binary frame decode as an int,
    # not as the map protocol.py's JSON fallback is documented to catch --
    # that fallback only fires when the primary msgpack decode actually
    # raises. JSON-over-binary is only reliably read when the transport hands
    # it to decode_client_frame as `str` (see test_decode_accepts_json_text_frame).
    raw = json.dumps({"event": "ping"}).encode("utf-8")
    with pytest.raises(ProtocolError):
        decode_client_frame(raw)


# -- malformed frames, every branch --


def test_decode_invalid_json_string_raises():
    with pytest.raises(ProtocolError):
        decode_client_frame("{not json")


def test_decode_neither_msgpack_nor_json_raises():
    with pytest.raises(ProtocolError):
        decode_client_frame(b"\xff\xfe\x00not a valid frame at all\x01")


def test_decode_non_dict_payload_raises():
    with pytest.raises(ProtocolError):
        decode_client_frame(ormsgpack.packb(["event", "ping"]))


def test_decode_missing_event_field_raises():
    with pytest.raises(ProtocolError):
        decode_client_frame(_msgpack({"text": "no event key"}))


def test_decode_event_field_not_a_string_raises():
    with pytest.raises(ProtocolError):
        decode_client_frame(_msgpack({"event": 123}))


def test_decode_unknown_event_raises():
    with pytest.raises(ProtocolError):
        decode_client_frame(_msgpack({"event": "teleport"}))


def test_decode_ttsparams_forbids_extra_fields():
    with pytest.raises(ProtocolError):
        decode_client_frame(_msgpack({"event": "start", "request": {"bogus_field": 1}}))


def test_decode_ttsparams_out_of_range_raises():
    with pytest.raises(ProtocolError):
        decode_client_frame(_msgpack({"event": "start", "request": {"top_p": 5.0}}))


def test_decode_ttsparams_bad_format_raises():
    with pytest.raises(ProtocolError):
        decode_client_frame(_msgpack({"event": "start", "request": {"format": "ogg"}}))


def test_decode_reference_audio_bad_base64_raises():
    payload = {"event": "start", "request": {"references": [{"audio": "not-base64!!", "text": "x"}]}}
    with pytest.raises(ProtocolError):
        decode_client_frame(_msgpack(payload))


def test_decode_reference_audio_missing_text_raises():
    payload = {"event": "start", "request": {"references": [{"audio": "aGVsbG8="}]}}
    with pytest.raises(ProtocolError):
        decode_client_frame(_msgpack(payload))


# -- server-side encoders --


def test_started_frame_roundtrip():
    frame = ormsgpack.unpackb(started_frame("sess-1", 24000, "wav", 12345))
    assert frame == {
        "event": "started",
        "session_id": "sess-1",
        "sample_rate": 24000,
        "format": "wav",
        "seed": 12345,
        "voice_id": None,  # no cloning voice in play
    }


def test_started_frame_reports_the_voice_in_use():
    frame = ormsgpack.unpackb(started_frame("sess-1", 24000, "wav", 1, "abc123"))
    assert frame["voice_id"] == "abc123"


def test_audio_frame_roundtrip():
    frame = ormsgpack.unpackb(audio_frame(b"\x01\x02\x03\x04", 12.5))
    assert frame["event"] == "audio"
    assert frame["audio"] == b"\x01\x02\x03\x04"
    assert frame["time"] == 12.5


def test_log_frame_roundtrip():
    frame = ormsgpack.unpackb(log_frame("hello"))
    assert frame == {"event": "log", "message": "hello"}


def test_finish_frame_roundtrip():
    frame = ormsgpack.unpackb(finish_frame("stop"))
    assert frame == {"event": "finish", "reason": "stop", "error": None}
    frame = ormsgpack.unpackb(finish_frame("error", "boom"))
    assert frame == {"event": "finish", "reason": "error", "error": "boom"}


def test_pong_frame_roundtrip():
    assert ormsgpack.unpackb(pong_frame()) == {"event": "pong"}


def test_max_new_tokens_defaults_to_the_whole_remaining_context():
    """0 means "as many frames as the context has room for". The old default of
    1024 was an invisible ceiling: a segment that hit it stopped mid-word and
    was indistinguishable from one that had finished on its own."""
    assert TTSParams().max_new_tokens == 0


def test_integer_bounds_are_actually_enforced():
    """`Annotated[int, conint(...)]` is a no-op under pydantic v2 - conint()
    builds a type, so as Annotated metadata it is silently ignored. Every
    integer bound in TTSParams was unenforced: `max_new_tokens=-1` gave
    `range(-2)`, i.e. an empty audio stream followed by a clean `finish`, which
    is indistinguishable from a segment the model chose not to speak."""
    assert TTSParams(max_new_tokens=512).max_new_tokens == 512
    for field, bad in [
        ("max_new_tokens", -1),
        ("max_new_tokens", 9000),
        ("chunk_length", -1),
        ("chunk_length", 50),
        ("chunk_length", 400),
        ("top_k", 0),
        ("top_k", 2000),
    ]:
        with pytest.raises(ValidationError):
            TTSParams(**{field: bad})


def test_server_defaults_fill_only_what_the_client_left_alone():
    """`TTSParams` carries its own defaults, so after validation there is no way
    to tell "200 because the operator configured it" from "200 because nobody
    said" - except `model_fields_set`, which holds exactly the keys the inbound
    frame contained. This is what makes TTS_CHUNK_LENGTH and TTS_DEFAULT_FORMAT
    mean anything; both used to be settings nothing read."""
    from app.protocol import with_server_defaults

    untouched = TTSParams()
    filled = with_server_defaults(untouched, chunk_length=300, format="mp3")
    assert filled.chunk_length == 300 and filled.format == "mp3"

    explicit = TTSParams.model_validate({"chunk_length": 150, "format": "flac"})
    kept = with_server_defaults(explicit, chunk_length=300, format="mp3")
    assert kept.chunk_length == 150 and kept.format == "flac"


def test_server_defaults_ignore_none_and_leave_the_object_alone():
    from app.protocol import with_server_defaults

    params = TTSParams()
    assert with_server_defaults(params, chunk_length=None) is params
    assert with_server_defaults(params) is params
