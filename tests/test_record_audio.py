import record_audio


def test_recording_defaults_to_stereo():
    assert record_audio.DEFAULT_CHANNELS == 2


def test_help_describes_stereo_recording():
    parser = record_audio.build_parser()
    assert "双声道" in parser.description
