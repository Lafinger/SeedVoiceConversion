import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from safetensors import SafetensorError
from transformers import WhisperConfig, WhisperModel

from api import hf_utils


class ModelCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cache = Path(temporary.name)
        self.repo = "test/model"
        self.commit = "a" * 40
        for patcher in (
            patch.object(hf_utils, "MODEL_CACHE_DIR", self.cache),
            patch.object(hf_utils.constants, "HF_HUB_OFFLINE", False),
            patch.dict(os.environ, {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        self.logger = contexts.enter_context(patch.object(hf_utils, "logger"))
        self.http = contexts.enter_context(patch("requests.Session.request", side_effect=AssertionError("unexpected HTTP")))
        self.socket = contexts.enter_context(patch("socket.socket.connect", side_effect=AssertionError("unexpected network")))

    def tearDown(self):
        self.http.assert_not_called()
        self.socket.assert_not_called()

    def put(self, filename, content=b"test", repo=None):
        repo_dir = self.cache / ("models--" + (repo or self.repo).replace("/", "--"))
        ref = repo_dir / "refs" / "main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(self.commit, encoding="utf-8")
        target = repo_dir / "snapshots" / self.commit / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
        return str(target)

    def test_custom_model_uses_cache_and_preserves_return_shapes(self):
        model = self.put("model.pth")
        config = self.put("config.yml")
        self.assertEqual(hf_utils.load_custom_model_from_hf(self.repo, "model.pth"), model)
        self.assertEqual(hf_utils.load_custom_model_from_hf(self.repo, "model.pth", "config.yml"), (model, config))

    def test_custom_model_downloads_only_missing_file_then_reuses_it(self):
        self.put("model.pth")
        downloaded = []

        def download(**kwargs):
            if kwargs["local_files_only"]:
                return hf_hub_download(**kwargs)
            downloaded.append(kwargs["filename"])
            return self.put(kwargs["filename"])

        with patch.object(hf_utils, "hf_hub_download", side_effect=download):
            first = hf_utils.load_custom_model_from_hf(self.repo, "model.pth", "config.yml")
            second = hf_utils.load_custom_model_from_hf(self.repo, "model.pth", "config.yml")
        self.assertEqual(first, second)
        self.assertEqual(downloaded, ["config.yml"])

    def test_custom_model_cold_cache_downloads_model_and_config(self):
        downloaded = []

        def download(**kwargs):
            if kwargs["local_files_only"]:
                return hf_hub_download(**kwargs)
            downloaded.append(kwargs["filename"])
            return self.put(kwargs["filename"])

        with patch.object(hf_utils, "hf_hub_download", side_effect=download):
            hf_utils.load_custom_model_from_hf(self.repo, "model.pth", "config.yml")
            hf_utils.load_custom_model_from_hf(self.repo, "model.pth", "config.yml")
        self.assertEqual(downloaded, ["model.pth", "config.yml"])

    def test_permission_error_is_not_a_cache_miss(self):
        failure = PermissionError("denied")
        with patch.object(hf_utils, "hf_hub_download", side_effect=failure) as download:
            with self.assertRaises(PermissionError) as raised:
                hf_utils.load_custom_model_from_hf(self.repo)
        self.assertIs(raised.exception, failure)
        download.assert_called_once()
        self.assertTrue(download.call_args.kwargs["local_files_only"])

    def test_download_failure_keeps_original_exception_and_logs_context(self):
        failure = ConnectionError("unavailable")
        with patch.object(hf_utils, "hf_hub_download", side_effect=[LocalEntryNotFoundError("missing"), failure]):
            with self.assertRaises(ConnectionError) as raised:
                hf_utils.load_custom_model_from_hf(self.repo, "model.pth")
        self.assertIs(raised.exception, failure)
        self.assertEqual(self.logger.exception.call_args.args[1:], (self.repo, "model.pth", self.cache))

    def test_pretrained_single_weight_formats_are_local(self):
        for weight in ("model.safetensors", "pytorch_model.bin"):
            with self.subTest(weight=weight):
                repo = "test/" + weight.replace(".", "-")
                self.put("config.json", "{}", repo)
                self.put(weight, repo=repo)
                loader = Mock()
                hf_utils.load_pretrained_from_hf(loader, repo, check_weights=True, torch_dtype="preserved")
                loader.assert_called_once_with(
                    repo, cache_dir=str(self.cache), local_files_only=True, torch_dtype="preserved",
                )

    def test_pretrained_cold_cache_downloads_then_loads_locally(self):
        def load(model_id, **kwargs):
            if not kwargs["local_files_only"]:
                self.put("config.json", "{}")
                self.put("model.safetensors")
            return "model"

        loader = Mock(side_effect=load)
        for _ in range(2):
            self.assertEqual(hf_utils.load_pretrained_from_hf(loader, self.repo, check_weights=True), "model")
        self.assertEqual([call.kwargs["local_files_only"] for call in loader.call_args_list], [False, True])

    def test_missing_required_component_files_allow_download(self):
        cases = (
            ("config", ["model.safetensors"], ("config.json",), True),
            ("weights", ["config.json"], ("config.json",), True),
            ("extractor", ["config.json"], ("preprocessor_config.json",), False),
            ("bigvgan", ["config.json"], ("config.json", "bigvgan_generator.pt"), False),
        )
        for name, present, required, check_weights in cases:
            with self.subTest(component=name):
                repo = "test/" + name
                for filename in present:
                    self.put(filename, "{}", repo)
                loader = Mock()
                hf_utils.load_pretrained_from_hf(
                    loader, repo, required_files=required, check_weights=check_weights,
                )
                loader.assert_called_once()
                self.assertFalse(loader.call_args.kwargs["local_files_only"])

    def test_sharded_weights_require_every_shard(self):
        for prefix in ("model.safetensors", "pytorch_model.bin"):
            with self.subTest(format=prefix):
                repo = "test/" + prefix.replace(".", "-")
                self.put("config.json", "{}", repo)
                self.put(prefix + ".index.json", json.dumps({"weight_map": {"a": "shard1", "b": "shard2"}}), repo)
                self.put("shard1", repo=repo)
                loader = Mock()
                hf_utils.load_pretrained_from_hf(loader, repo, check_weights=True)
                self.assertFalse(loader.call_args.kwargs["local_files_only"])
                self.put("shard2", repo=repo)
                hf_utils.load_pretrained_from_hf(loader, repo, check_weights=True)
                self.assertTrue(loader.call_args.kwargs["local_files_only"])

    def test_invalid_shard_index_does_not_trigger_download(self):
        self.put("config.json", "{}")
        for contents in ("broken json", "{}", "[]", '{"weight_map": {}}', '{"weight_map": {"a": 1}}'):
            with self.subTest(contents=contents):
                self.put("model.safetensors.index.json", contents)
                loader = Mock()
                with self.assertRaises(ValueError):
                    hf_utils.load_pretrained_from_hf(loader, self.repo, check_weights=True)
                loader.assert_not_called()

    def test_load_errors_are_not_retried(self):
        self.put("config.json", "{}")
        self.put("model.safetensors")
        for failure in (RuntimeError("CUDA out of memory"), OSError("corrupt weights"), ValueError("bad config")):
            with self.subTest(failure=failure):
                loader = Mock(side_effect=failure)
                with self.assertRaises(type(failure)) as raised:
                    hf_utils.load_pretrained_from_hf(loader, self.repo, check_weights=True)
                self.assertIs(raised.exception, failure)
                loader.assert_called_once()
                self.assertTrue(loader.call_args.kwargs["local_files_only"])

    def test_real_transformers_rejects_corrupt_weights_without_network(self):
        self.put("config.json", WhisperConfig().to_json_string())
        corrupt = self.put("model.safetensors", b"corrupt")
        loader = Mock(wraps=WhisperModel.from_pretrained)
        with self.assertRaises(SafetensorError):
            hf_utils.load_pretrained_from_hf(loader, self.repo, check_weights=True)
        loader.assert_called_once()
        self.assertEqual(Path(corrupt).read_bytes(), b"corrupt")

    def test_real_transformers_rejects_invalid_config_without_network(self):
        self.put("config.json", "broken json")
        self.put("model.safetensors")
        loader = Mock(wraps=WhisperModel.from_pretrained)
        with self.assertRaises(OSError):
            hf_utils.load_pretrained_from_hf(loader, self.repo, check_weights=True)
        loader.assert_called_once()

    def test_explicit_offline_environment_blocks_missing_files(self):
        for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            with self.subTest(variable=variable), patch.dict(os.environ, {variable: "1"}):
                loader = Mock()
                with self.assertRaisesRegex(LocalEntryNotFoundError, "离线模式"):
                    hf_utils.load_pretrained_from_hf(loader, self.repo, check_weights=True)
                loader.assert_not_called()
                with self.assertRaisesRegex(LocalEntryNotFoundError, "离线模式"):
                    hf_utils.load_custom_model_from_hf(self.repo)

    def test_explicit_offline_environment_allows_complete_cache(self):
        self.put("config.json", "{}")
        self.put("model.safetensors")
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}):
            loader = Mock()
            hf_utils.load_pretrained_from_hf(loader, self.repo, check_weights=True)
        self.assertTrue(loader.call_args.kwargs["local_files_only"])

    def test_explicit_local_paths_never_fall_back_to_hub(self):
        with patch.object(hf_utils, "hf_hub_download") as download:
            loader = Mock()
            hf_utils.load_pretrained_from_hf(loader, self.cache)
            self.assertTrue(loader.call_args.kwargs["local_files_only"])
            for missing in (self.cache / "absent", "./absent-model", ".\\absent-model"):
                with self.subTest(path=missing), self.assertRaises(FileNotFoundError):
                    hf_utils.load_pretrained_from_hf(loader, missing)
            download.assert_not_called()
            loader.assert_called_once()


class CacheLocationTests(unittest.TestCase):
    def test_cache_location_does_not_depend_on_working_directory(self):
        root = Path(__file__).resolve().parents[1]
        script = f"import sys; sys.path.insert(0, {str(root)!r}); from api.hf_utils import MODEL_CACHE_DIR; print(MODEL_CACHE_DIR)"
        for cwd in (root, root / "api"):
            with self.subTest(cwd=cwd):
                result = subprocess.run([sys.executable, "-c", script], cwd=cwd, capture_output=True, text=True, check=True)
                self.assertEqual(Path(result.stdout.strip()), root / "checkpoints")


if __name__ == "__main__":
    unittest.main()
