from types import ModuleType, SimpleNamespace

from conftest import FakeBox, FakeImage


def test_associate_direct_overlap_no_helmet_and_plate(load_cli_module):
    detect, _ = load_cli_module("detect")
    no_helmets = [{"index": 0, "cls_name": "No-Helmet", "conf": 0.9, "xyxy": (10, 10, 120, 140)}]
    plates = [{"index": 1, "cls_name": "Number-Plate", "conf": 0.85, "xyxy": (40, 100, 100, 130)}]

    associations = detect.associate_plates_to_violations(no_helmets, [], plates)
    assert len(associations) == 1
    assert len(associations[0]["plates"]) == 1
    assert associations[0]["plates"][0]["index"] == 1


def test_associate_plate_via_rider_box_without_direct_head_overlap(load_cli_module):
    detect, _ = load_cli_module("detect")
    no_helmets = [{"index": 0, "cls_name": "No-Helmet", "conf": 0.9, "xyxy": (120, 90, 180, 140)}]
    riders = [{"index": 1, "cls_name": "Rider", "conf": 0.92, "xyxy": (80, 80, 260, 420)}]
    plates = [{"index": 2, "cls_name": "Number-Plate", "conf": 0.88, "xyxy": (130, 320, 200, 360)}]

    associations = detect.associate_plates_to_violations(no_helmets, riders, plates)
    assert len(associations) == 1
    assert associations[0]["rider"]["index"] == 1
    assert [p["index"] for p in associations[0]["plates"]] == [2]


def test_associate_excludes_compliant_rider_and_unrelated_plates(load_cli_module):
    detect, _ = load_cli_module("detect")
    # Compliant rider on left (x=10..190), violating rider on right (x=180..360)
    riders = [
        {"index": 0, "cls_name": "Rider", "conf": 0.95, "xyxy": (10, 50, 190, 400)},
        {"index": 1, "cls_name": "Rider", "conf": 0.94, "xyxy": (180, 50, 360, 400)},
    ]
    no_helmets = [{"index": 2, "cls_name": "No-Helmet", "conf": 0.91, "xyxy": (240, 60, 300, 110)}]
    plates = [
        {"index": 3, "cls_name": "Number-Plate", "conf": 0.89, "xyxy": (70, 310, 130, 345)},
        {"index": 4, "cls_name": "Number-Plate", "conf": 0.87, "xyxy": (240, 310, 300, 345)},
        {"index": 5, "cls_name": "Number-Plate", "conf": 0.86, "xyxy": (500, 300, 580, 340)},
    ]

    associations = detect.associate_plates_to_violations(no_helmets, riders, plates)
    assert len(associations) == 1
    assert [p["index"] for p in associations[0]["plates"]] == [4]


def test_no_violations_yields_no_associated_plates(load_cli_module):
    detect, _ = load_cli_module("detect")
    riders = [{"index": 0, "cls_name": "Rider", "conf": 0.95, "xyxy": (50, 50, 250, 400)}]
    plates = [{"index": 1, "cls_name": "Number-Plate", "conf": 0.9, "xyxy": (100, 320, 170, 355)}]

    assert detect.associate_plates_to_violations([], riders, plates) == []


def test_run_pipeline_saves_plate_crops_and_runs_ocr_only_on_violations(tmp_path, load_cli_module):
    names = {0: "Helmet", 1: "No-Helmet", 2: "Rider", 3: "Number-Plate"}
    frame = FakeImage(480, 640)
    boxes = [
        # Compliant rider + plate (should NOT be saved or OCR'd)
        FakeBox(0, 0.95, (40, 40, 100, 90)),
        FakeBox(2, 0.93, (20, 30, 150, 350)),
        FakeBox(3, 0.88, (50, 270, 120, 310)),
        # Violating rider + plate (SHOULD be saved to crops/violations & crops/plates and OCR'd)
        FakeBox(1, 0.91, (320, 50, 380, 100)),
        FakeBox(2, 0.94, (280, 40, 440, 380)),
        FakeBox(3, 0.90, (330, 290, 400, 330)),
    ]
    fake_result = SimpleNamespace(orig_img=frame, boxes=boxes)

    class FakeYOLO:
        def __init__(self, weights):
            self.weights = weights
            self.names = names

        def predict(self, **kwargs):
            return [fake_result]

    ocr_calls = []

    class FakeReader:
        def __init__(self, langs, gpu=False):
            self.langs = langs
            self.gpu = gpu

        def readtext(self, img):
            ocr_calls.append(img.shape)
            return [([[0, 0], [1, 0], [1, 1], [0, 1]], "KA01AB1234", 0.92)]

    fake_easyocr = ModuleType("easyocr")
    fake_easyocr.Reader = FakeReader
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))

    detect, fake_cv2 = load_cli_module(
        "detect",
        yolo_factory=FakeYOLO,
        easyocr_mod=fake_easyocr,
        torch_mod=fake_torch,
    )

    opt = detect.parse_opt(
        ["--source", "test.jpg", "--ocr", "--save-crop", "--output", str(tmp_path)]
    )
    summary = detect.run_pipeline(opt)

    assert summary["violations"] == 1
    assert summary["plates"] == ["KA01AB1234"]
    assert len(ocr_calls) == 1
    assert ocr_calls[0] == (40, 70, 3)

    written_paths = [p.replace("\\", "/") for p, _ in fake_cv2.written]
    assert any("crops/violations/violation_1_3.jpg" in p for p in written_paths)
    assert any("crops/plates/plate_1_5.jpg" in p for p in written_paths)
    assert not any("plate_1_2.jpg" in p for p in written_paths)


def test_run_pipeline_skips_empty_crops_before_imwrite(tmp_path, load_cli_module):
    names = {0: "No-Helmet", 1: "Number-Plate"}
    frame = FakeImage(100, 100)
    # Out-of-bounds / degenerate boxes that yield 0-size crops
    boxes = [
        FakeBox(0, 0.90, (120, 120, 180, 180)),
        FakeBox(1, 0.85, (130, 130, 170, 170)),
    ]
    fake_result = SimpleNamespace(orig_img=frame, boxes=boxes)

    class FakeYOLO:
        def __init__(self, weights):
            self.names = names

        def predict(self, **kwargs):
            return [fake_result]

    ocr_calls = []
    fake_easyocr = ModuleType("easyocr")
    fake_easyocr.Reader = lambda *a, **kw: SimpleNamespace(
        readtext=lambda img: ocr_calls.append(img) or []
    )

    detect, fake_cv2 = load_cli_module(
        "detect",
        yolo_factory=FakeYOLO,
        easyocr_mod=fake_easyocr,
        torch_mod=None,
    )

    opt = detect.parse_opt(
        ["--source", "test.jpg", "--ocr", "--save-crop", "--output", str(tmp_path)]
    )
    summary = detect.run_pipeline(opt)

    assert summary["violations"] == 1
    assert fake_cv2.written == []
    assert ocr_calls == []


def test_easyocr_gpu_detection_handles_devices_and_missing_or_mocked_torch(load_cli_module):
    detect_cuda, _ = load_cli_module(
        "detect",
        torch_mod=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    assert detect_cuda._should_use_easyocr_gpu("") is True
    assert detect_cuda._should_use_easyocr_gpu("cpu") is False
    assert detect_cuda._should_use_easyocr_gpu("0") is True

    detect_no_cuda, _ = load_cli_module(
        "detect",
        torch_mod=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    assert detect_no_cuda._should_use_easyocr_gpu("") is False

    detect_missing_torch, _ = load_cli_module("detect", torch_mod=None)
    assert detect_missing_torch._should_use_easyocr_gpu("") is False

    detect_partial_mock, _ = load_cli_module("detect", torch_mod=SimpleNamespace())
    assert detect_partial_mock._should_use_easyocr_gpu("") is False
