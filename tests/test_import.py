def test_version():
    from data4d import __version__
    from packaging.version import parse

    assert parse(__version__) >= parse("0.1.0")
