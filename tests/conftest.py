import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class FakeScalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value

    def __int__(self):
        return int(self._value)

    def __float__(self):
        return float(self._value)


class FakeCoords:
    def __init__(self, coords):
        self._coords = list(coords)

    def cpu(self):
        return self

    def numpy(self):
        return self

    def astype(self, _dtype):
        return self

    def __iter__(self):
        return iter(self._coords)


class FakeBox:
    def __init__(self, cls_id: int, conf: float, xyxy):
        self.cls = [FakeScalar(cls_id)]
        self.conf = [FakeScalar(conf)]
        self.xyxy = [FakeCoords(xyxy)]


class FakeImage:
    def __init__(self, height: int, width: int, channels: int = 3):
        self.height = height
        self.width = width
        self.channels = channels

    @property
    def shape(self):
        return (self.height, self.width, self.channels)

    @property
    def size(self):
        return self.height * self.width * self.channels

    def __getitem__(self, item):
        row_slice, col_slice = item
        h = max(0, (row_slice.stop or 0) - (row_slice.start or 0))
        w = max(0, (col_slice.stop or 0) - (col_slice.start or 0))
        return FakeImage(h, w, self.channels)


@pytest.fixture
def load_cli_module(monkeypatch):
    def _loader(
        module_name: str,
        *,
        yolo_factory=None,
        cv2_mod=None,
        easyocr_mod=None,
        torch_mod="UNSET",
    ):
        if cv2_mod is None:
            written = []

            def fake_imwrite(path, img):
                written.append((path, img))
                return True

            cv2_mod = ModuleType("cv2")
            cv2_mod.imwrite = fake_imwrite
            cv2_mod.written = written

        monkeypatch.setitem(sys.modules, "cv2", cv2_mod)

        ultralytics_mod = ModuleType("ultralytics")
        if yolo_factory is None:
            yolo_factory = lambda *a, **kw: SimpleNamespace(names={}, predict=lambda **k: [])
        ultralytics_mod.YOLO = yolo_factory
        monkeypatch.setitem(sys.modules, "ultralytics", ultralytics_mod)

        if easyocr_mod is not None:
            monkeypatch.setitem(sys.modules, "easyocr", easyocr_mod)

        if torch_mod != "UNSET":
            monkeypatch.setitem(sys.modules, "torch", torch_mod)

        monkeypatch.delitem(sys.modules, module_name, raising=False)
        mod = importlib.import_module(module_name)
        return mod, cv2_mod

    return _loader
