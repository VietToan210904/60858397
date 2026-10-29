"""CPU-only synthetic safety fixtures; no benchmark data or network requests."""
import importlib.util
import io
from pathlib import Path
import stat
import tarfile
import zipfile

import pytest


def module(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


gso = module("acquire_gso_reference")
omni = module("acquire_omni_reference")


@pytest.mark.parametrize("name", ["../escape.obj", "/absolute.obj", "C:/escape.obj", "..\\escape.obj"])
def test_zip_traversal_refused(tmp_path, name):
    archive = tmp_path / "test.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(name, b"synthetic")
    with pytest.raises(ValueError, match="Unsafe ZIP"):
        gso.safe_zip(archive, tmp_path / "output")


def test_zip_symlink_refused(tmp_path):
    archive = tmp_path / "test.zip"
    info = zipfile.ZipInfo("mesh.obj")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(info, "elsewhere")
    with pytest.raises(ValueError, match="links/devices"):
        gso.safe_zip(archive, tmp_path / "output")


def test_zip_expansion_cap_and_existing_integrity(tmp_path):
    archive = tmp_path / "test.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("meshes/model.obj", b"synthetic fixture")
        z.writestr("do_not_run.exe", b"not executable data")
    with pytest.raises(ValueError, match="expansion cap"):
        gso.safe_zip(archive, tmp_path / "small", remaining=2)
    result = gso.safe_zip(archive, tmp_path / "output")
    assert result["skipped_non_data_members"] == ["do_not_run.exe"]
    assert not (tmp_path / "output/do_not_run.exe").exists()
    (tmp_path / "output/meshes/model.obj").write_bytes(b"changed")
    with pytest.raises(ValueError, match="no overwrite"):
        gso.safe_zip(archive, tmp_path / "output")


def test_tar_links_and_expansion_refused(tmp_path):
    archive = tmp_path / "test.tar.gz"
    with tarfile.open(archive, "w:gz") as t:
        info = tarfile.TarInfo("link.obj")
        info.type = tarfile.SYMTYPE
        info.linkname = "../outside"
        t.addfile(info)
    with pytest.raises(ValueError, match="links/devices"):
        omni.safe_extract(archive, tmp_path / "output")
    with tarfile.open(archive, "w:gz") as t:
        info = tarfile.TarInfo("mesh.obj")
        info.size = 10
        t.addfile(info, io.BytesIO(b"0123456789"))
    with pytest.raises(ValueError, match="expansion cap"):
        omni.safe_extract(archive, tmp_path / "output", remaining=5)


def test_category_does_not_relabel_pencil_cases_or_other_owners():
    source = {"owner": "GoogleResearch", "license_id": 2, "private": False, "name": "Pencil_Case", "categories": ["Bag"]}
    assert gso.category(source) is None
    assert gso.category({**source, "name": "School_Backpack"}) == "backpack"
    assert gso.category({**source, "name": "Shoe", "categories": ["Shoe"]}) == "shoe"
    assert gso.category({**source, "owner": "someone_else", "categories": ["Shoe"]}) is None


class Response:
    status_code = 200
    def __init__(self, body, content_type="text/html"):
        self.body = body
        self.headers = {"Content-Type": content_type}
    def raise_for_status(self):
        pass
    def iter_content(self, size):
        yield self.body
    def close(self):
        pass


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)


def notice(file_id):
    return ("Google Drive can't scan this file because it is too large."
            '<form action="https://drive.usercontent.google.com/download">'
            f'<input name="id" value="{file_id}"><input name="export" value="download">'
            '<input name="confirm" value="t"></form>').encode()


def test_only_ordinary_large_file_confirmation_is_accepted(tmp_path):
    item = {"file_id": "official", "size_bytes": 3}
    session = Session([Response(notice("official")), Response(b"zip", "application/gzip")])
    response, evidence = omni.public_response(session, item, omni.Ledger(tmp_path / "ledger.json"), 0)
    assert response.body == b"zip"
    assert evidence["type"] == "large_file_not_scanned"
    assert len(session.calls) == 2


@pytest.mark.parametrize("body", [b"This file is infected", b"This file violates abuse policy", notice("different_file")])
def test_security_findings_and_changed_identity_refused(tmp_path, body):
    session = Session([Response(body)])
    with pytest.raises(ValueError):
        omni.public_response(session, {"file_id": "official", "size_bytes": 3}, omni.Ledger(tmp_path / "ledger.json"), 0)
    assert len(session.calls) == 1


def test_quota_is_not_bypassed(tmp_path):
    session = Session([Response(notice("official")), Response(b"Google Drive - Quota exceeded")])
    with pytest.raises(ValueError, match="quota exceeded"):
        omni.public_response(session, {"file_id": "official", "size_bytes": 3}, omni.Ledger(tmp_path / "ledger.json"), 0)
    assert len(session.calls) == 2
