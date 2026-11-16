"""Offline tests for the desktop exe: subprocess console hygiene and the
first-run desktop shortcut choice.

Follows the repository's harness conventions: no real CLI processes are
launched (Popen is replaced), no network, no account data. The spawn probe
evidence lives in launcher/test_app.ps1 and docs/windows.md.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
(ROOT / "tmp").mkdir(exist_ok=True)

from webapp import cli_runner, spawn  # noqa: E402
from launcher import launcher  # noqa: E402


class SpawnTests(unittest.TestCase):
    def test_no_window_flag(self):
        if os.name == "nt":
            self.assertEqual(spawn.NO_WINDOW, subprocess.CREATE_NO_WINDOW)
        else:
            self.assertEqual(spawn.NO_WINDOW, 0)

    def test_quiet_flags_include_the_extras_on_windows(self):
        extra = subprocess.CREATE_NEW_PROCESS_GROUP
        flags = spawn.quiet_flags(extra)
        if os.name == "nt":
            self.assertTrue(flags & spawn.NO_WINDOW)
            self.assertTrue(flags & extra)
        else:
            self.assertEqual(flags, 0)

    def test_platform_launch_suppresses_the_console(self):
        captured = []

        class FakePopen:
            def __init__(self, args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                captured.append(self)

        original = subprocess.Popen
        subprocess.Popen = FakePopen
        try:
            command = cli_runner.Command(str(Path("C:\\") / "fake" / "claude.cmd"),
                                         ["mcp", "list"])
            process = cli_runner.platform_launch(command, ".", {})
        finally:
            subprocess.Popen = original
        self.assertIs(process, captured[0])
        self.assertEqual(captured[0].args,
                         ["cmd", "/c", str(Path("C:\\") / "fake" / "claude.cmd"),
                          "mcp", "list"])
        if os.name == "nt":
            self.assertTrue(captured[0].kwargs["creationflags"] & spawn.NO_WINDOW)

    def test_plain_subprocess_run_sites_carry_the_flag(self):
        # The curl fallback in transport is the only run() that pipes network
        # output; it is covered by test_cloud.py. This test pins the helper.
        self.assertEqual(spawn.quiet_flags(0), spawn.NO_WINDOW)


class ShortcutChoiceTests(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            path = Path(tmp) / "config.json"
            self.assertEqual(launcher.choice_made(path), (False, False))
            launcher.save_choice(True, path)
            self.assertEqual(launcher.choice_made(path), (True, True))
            launcher.save_choice(False, path)
            self.assertEqual(launcher.choice_made(path), (True, False))

    def test_corrupt_config_counts_as_not_made_and_is_repaired(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("not json at all", encoding="utf-8")
            self.assertEqual(launcher.choice_made(path), (False, False))
            launcher.save_choice(True, path)
            self.assertEqual(launcher.choice_made(path), (True, True))

    def test_other_config_fields_survive_the_choice(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            path = Path(tmp) / "config.json"
            path.write_text('{"workspaceRoot": "C:\\\\repo", "port": 8765, '
                            '"token": "t", "access": "lan"}', encoding="utf-8")
            launcher.save_choice(True, path)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["workspaceRoot"], "C:\\repo")
            self.assertEqual(data["port"], 8765)
            self.assertEqual(data["token"], "t")
            self.assertEqual(data["access"], "lan")
            self.assertEqual((data["shortcutChoiceMade"], data["desktopShortcut"]),
                             (True, True))

    @unittest.skipUnless(os.name == "nt", "Windows desktop")
    def test_desktop_dir_exists(self):
        self.assertTrue(launcher.desktop_dir().is_dir())


if __name__ == "__main__":
    unittest.main()
