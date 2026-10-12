import sys
from types import SimpleNamespace

import autoclipper.ingest as ingest


def test_finalized_youtube_options_retry_extractor_requests_without_disabling_tls():
    options = ingest._finalize_opts()
    assert options["retries"] >= 1
    assert options["extractor_retries"] >= 5
    assert "nocheckcertificate" not in options


def test_transient_network_detection_targets_tls_eof_not_provider_errors():
    tls_error = RuntimeError(
        "Unable to download API page: TLS/SSL connection has been closed (EOF) (_ssl.c:992)"
    )
    curl_tls_error = RuntimeError(
        "curl: (35) BoringSSL SSL_connect: Connection closed abruptly "
        "(SSL_ERROR_SYSCALL; error queue empty)"
    )
    assert ingest._is_transient_network_error(tls_error)
    assert ingest._is_transient_network_error(curl_tls_error)
    assert not ingest._is_transient_network_error(RuntimeError("HTTP Error 403: Forbidden"))


def test_youtube_download_retries_a_transient_tls_eof(monkeypatch, tmp_path):
    calls = {"count": 0}
    output_path = tmp_path / "demo.mp4"
    tls_error = RuntimeError(
        "Unable to download API page: TLS/SSL connection has been closed (EOF) (_ssl.c:992)"
    )

    class FakeYoutubeDL:
        def __init__(self, _options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def extract_info(self, _url, download=True):
            assert download is True
            calls["count"] += 1
            if calls["count"] == 1:
                raise tls_error
            output_path.write_bytes(b"synthetic video")
            return {"title": "Retry demo", "filepath": str(output_path)}

        def prepare_filename(self, info):
            return info["filepath"]

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))
    monkeypatch.setattr(ingest, "_write_cookies", lambda *_args: None)
    monkeypatch.setattr(ingest.time, "sleep", lambda _seconds: None)

    path, title = ingest._download_youtube(
        "https://www.youtube.com/watch?v=XTBHkR_tmKo",
        str(tmp_path),
    )

    assert calls["count"] == 2
    assert path == str(output_path)
    assert title


def test_youtube_download_uses_browser_tls_fallback_after_repeated_tls_eof(monkeypatch, tmp_path):
    chrome_target = object()
    output_path = tmp_path / "browser-fallback.mp4"
    options_seen = []

    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options
            options_seen.append(options)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def extract_info(self, _url, download=True):
            assert download is True
            if self.options.get("impersonate") is not chrome_target:
                raise RuntimeError("TLS/SSL connection has been closed (EOF)")
            output_path.write_bytes(b"synthetic video")
            return {"title": "Fallback demo", "filepath": str(output_path)}

        def prepare_filename(self, info):
            return info["filepath"]

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))
    monkeypatch.setattr(ingest, "_write_cookies", lambda *_args: None)
    monkeypatch.setattr(ingest, "_chrome_impersonation_target", lambda: chrome_target)
    monkeypatch.setattr(ingest.time, "sleep", lambda _seconds: None)

    path, title = ingest._download_youtube(
        "https://www.youtube.com/watch?v=XTBHkR_tmKo",
        str(tmp_path),
    )

    assert path == str(output_path)
    assert title
    assert any(options.get("impersonate") is chrome_target for options in options_seen)
