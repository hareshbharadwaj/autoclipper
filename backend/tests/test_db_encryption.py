import os
import stat

import pytest
from cryptography.fernet import Fernet

from autoclipper import db


def test_local_fernet_key_is_persisted_with_private_permissions(tmp_path):
    key_path = tmp_path / "autoclipper.fernet-key"

    first_key = db._persistent_fernet_key(str(key_path))
    second_key = db._persistent_fernet_key(str(key_path))
    ciphertext = Fernet(first_key).encrypt(b"synthetic test value")

    assert first_key == second_key
    assert Fernet(second_key).decrypt(ciphertext) == b"synthetic test value"
    assert stat.S_IMODE(os.stat(key_path).st_mode) == 0o600


def test_corrupt_persistent_fernet_key_fails_closed(tmp_path):
    key_path = tmp_path / "autoclipper.fernet-key"
    key_path.write_text("not-a-fernet-key", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unreadable or invalid"):
        db._persistent_fernet_key(str(key_path))
