from types import SimpleNamespace


def test_predict_save_and_no_save_flags_and_none_boxes_guard(load_cli_module, capsys):
    predict_calls = []

    class FakeYOLO:
        def __init__(self, weights):
            self.names = {0: "Helmet"}

        def predict(self, **kwargs):
            predict_calls.append(kwargs)
            return [SimpleNamespace(boxes=None)]

    predict, _ = load_cli_module("predict", yolo_factory=FakeYOLO)

    args_default = predict.parse_args([])
    assert args_default.save is True

    args_no_save = predict.parse_args(["--no-save", "--source", "img.jpg"])
    assert args_no_save.save is False

    results = predict.main(["--no-save", "--source", "img.jpg"])
    assert len(results) == 1
    assert predict_calls[0]["save"] is False
    assert "0 detections found" in capsys.readouterr().out


def test_predict_handles_empty_results_list(load_cli_module, capsys):
    class FakeYOLO:
        def __init__(self, weights):
            self.names = {}

        def predict(self, **kwargs):
            return []

    predict, _ = load_cli_module("predict", yolo_factory=FakeYOLO)
    results = predict.main(["--source", "img.jpg"])
    assert results == []
    assert "No prediction results returned." in capsys.readouterr().out


def test_train_none_results_guard(load_cli_module, capsys):
    class FakeYOLO:
        def __init__(self, model):
            self.model = model

        def train(self, **kwargs):
            return None

    train, _ = load_cli_module("train", yolo_factory=FakeYOLO)
    res = train.main(["--data", "data.yaml", "--epochs", "1"])
    assert res is None
    assert "without returning result metadata" in capsys.readouterr().out


def test_train_prints_save_dir_when_present(load_cli_module, capsys):
    class FakeYOLO:
        def __init__(self, model):
            self.model = model

        def train(self, **kwargs):
            return SimpleNamespace(save_dir="runs/train/road_safety_model")

    train, _ = load_cli_module("train", yolo_factory=FakeYOLO)
    res = train.main(["--data", "data.yaml", "--epochs", "1"])
    assert res.save_dir == "runs/train/road_safety_model"
    assert "runs/train/road_safety_model/weights/best.pt" in capsys.readouterr().out


def test_val_none_metrics_guard(load_cli_module, capsys):
    class FakeYOLO:
        def __init__(self, weights):
            self.weights = weights

        def val(self, **kwargs):
            return None

    val, _ = load_cli_module("val", yolo_factory=FakeYOLO)
    metrics = val.main(["--data", "data.yaml"])
    assert metrics is None
    assert "without box metrics" in capsys.readouterr().out
