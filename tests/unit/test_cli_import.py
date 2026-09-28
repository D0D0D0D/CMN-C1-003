"""Regression coverage for the Marketplace CLI module boundary."""


def test_cli_module_imports_on_this_sdk_version():
    """Marketplace runner は Pod 起動時に cli.py を import する。
    ここが落ちると業務リクエスト以前に Pod が死ぬ。
    """
    import importlib.util
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("cli_import_probe", repo / "cli.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cli_import_probe"] = module  # @dataclass の型解決に必要
    spec.loader.exec_module(module)  # ここで ImportError なら失敗
    assert hasattr(module, "MarketplaceGraph")
