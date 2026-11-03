"""No-game checks for managed-copy-only optional offline Gradle launching."""
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import envs.runtime as runtime


SCRIPT = ('#!/bin/bash\n# cmd="./gradlew runClient ignored comment"\n'
          '    cmd="./gradlew runClient --stacktrace -PrunDir=$runDir"\n'
          'else_cmd="java some.jar"\n')


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        runtime.configure_minedojo_runtime(False)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root/"installed"/"Minecraft"
        self.source.mkdir(parents=True)
        self.source_script = self.source/"launchClient.sh"
        self.source_script.write_text(SCRIPT)
        self.instance_root = self.root/"instance"
        self.minecraft = self.instance_root/"Minecraft"
        self.minecraft.mkdir(parents=True)
        self.script = self.minecraft/"launchClient.sh"
        self.script.write_text(SCRIPT)
        self.script.chmod(0o750)
        class Instance:
            def __init__(inner):
                inner.instance_dir = self.instance_root
                inner.calls = []
            def _launch_minecraft(inner, port, minecraft_dir, replaceable=True):
                inner.calls.append((port, Path(minecraft_dir), replaceable,
                                    (Path(minecraft_dir)/"launchClient.sh").read_text()))
                return "started"
        self.instance_class = Instance
        self.instance = Instance()
        self.loader = mock.patch.object(runtime, "_load_minecraft_runtime",
                                         return_value=(Instance, self.source))
        self.loader.start()

    def tearDown(self):
        runtime.configure_minedojo_runtime(False)
        self.loader.stop()
        self.temporary.cleanup()

    def test_default_false_does_not_import_or_modify(self):
        before = self.script.read_bytes()
        method = self.instance_class._launch_minecraft
        with mock.patch.object(runtime, "_load_minecraft_runtime", side_effect=AssertionError("must not import")):
            runtime.configure_minedojo_runtime()
        self.assertIs(self.instance_class._launch_minecraft, method)
        self.assertEqual(self.script.read_bytes(), before)

    def test_only_managed_copy_changes_and_permissions_preserved(self):
        source = self.source_script.read_bytes()
        mode = stat.S_IMODE(self.script.stat().st_mode)
        runtime.configure_minedojo_runtime(True)
        self.assertEqual(self.instance._launch_minecraft(12345, self.minecraft, False), "started")
        expected = SCRIPT.replace('cmd="./gradlew runClient --stacktrace',
                                  'cmd="./gradlew --offline runClient --stacktrace')
        self.assertEqual(self.script.read_text(), expected)
        self.assertEqual(self.source_script.read_bytes(), source)
        self.assertEqual(stat.S_IMODE(self.script.stat().st_mode), mode)
        self.assertEqual(self.instance.calls[-1][:3], (12345, self.minecraft, False))

    def test_repeated_configuration_and_launch_are_idempotent(self):
        runtime.configure_minedojo_runtime(True)
        method = self.instance_class._launch_minecraft
        self.instance._launch_minecraft(1, self.minecraft)
        once = self.script.read_bytes()
        runtime.configure_minedojo_runtime(True)
        self.assertIs(self.instance_class._launch_minecraft, method)
        self.instance._launch_minecraft(2, self.minecraft)
        self.assertEqual(self.script.read_bytes(), once)
        self.assertEqual(self.script.read_text().count("--offline"), 1)

    def test_false_restores_only_our_edit_before_relaunch(self):
        runtime.configure_minedojo_runtime(True)
        self.instance._launch_minecraft(1, self.minecraft)
        runtime.configure_minedojo_runtime(False)
        self.instance._launch_minecraft(2, self.minecraft)
        self.assertEqual(self.script.read_text(), SCRIPT)
        self.assertEqual(self.instance.calls[-1][-1], SCRIPT)

    def test_false_leaves_new_managed_copies_unchanged(self):
        runtime.configure_minedojo_runtime(True)
        runtime.configure_minedojo_runtime(False)
        self.instance._launch_minecraft(2, self.minecraft)
        self.assertEqual(self.script.read_text(), SCRIPT)

    def test_existing_offline_is_not_duplicated_or_removed(self):
        original = SCRIPT.replace("./gradlew runClient --stacktrace", "./gradlew --offline runClient --stacktrace")
        self.script.write_text(original)
        runtime.configure_minedojo_runtime(True)
        self.instance._launch_minecraft(1, self.minecraft)
        runtime.configure_minedojo_runtime(False)
        self.instance._launch_minecraft(2, self.minecraft)
        self.assertEqual(self.script.read_text(), original)

    def test_path_mismatch_is_rejected_without_modification(self):
        other = self.root/"other"/"Minecraft"
        other.mkdir(parents=True)
        target = other/"launchClient.sh";target.write_text(SCRIPT)
        runtime.configure_minedojo_runtime(True)
        with self.assertRaisesRegex(RuntimeError, "managed Minecraft copy.*cached"):
            self.instance._launch_minecraft(1, other)
        self.assertEqual(target.read_text(), SCRIPT)
        self.assertFalse(self.instance.calls)

    def test_installed_source_is_rejected_even_with_matching_instance_attr(self):
        self.instance.instance_dir = self.source.parent
        runtime.configure_minedojo_runtime(True)
        with self.assertRaisesRegex(RuntimeError, "installed MineDojo source"):
            self.instance._launch_minecraft(1, self.source)
        self.assertEqual(self.source_script.read_text(), SCRIPT)

    @unittest.skipIf(os.name == "nt", "WSL tests exercise POSIX symlink protection")
    def test_directory_and_script_symlinks_cannot_escape(self):
        outside = self.root/"outside";outside.mkdir()
        external = outside/"launchClient.sh";external.write_text(SCRIPT)
        self.script.unlink();self.script.symlink_to(external)
        runtime.configure_minedojo_runtime(True)
        with self.assertRaisesRegex(RuntimeError, "regular file"):
            self.instance._launch_minecraft(1, self.minecraft)
        self.assertEqual(external.read_text(), SCRIPT)
        self.script.unlink();self.minecraft.rmdir();self.minecraft.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "escapes instance_dir"):
            self.instance._launch_minecraft(1, self.minecraft)
        self.assertEqual(external.read_text(), SCRIPT)

    @unittest.skipIf(os.name == "nt", "WSL tests exercise POSIX hardlink protection")
    def test_hardlinked_installed_script_is_rejected(self):
        self.script.unlink();os.link(self.source_script, self.script)
        runtime.configure_minedojo_runtime(True)
        with self.assertRaisesRegex(RuntimeError, "hardlinked"):
            self.instance._launch_minecraft(1, self.minecraft)
        self.assertEqual(self.source_script.read_text(), SCRIPT)

    def test_missing_directory_or_command_gives_cache_dependency_hint(self):
        runtime.configure_minedojo_runtime(True)
        with self.assertRaisesRegex(RuntimeError, "cached on this machine"):
            self.instance._launch_minecraft(1, self.root/"absent")
        self.script.write_text('#!/bin/bash\n# cmd="./gradlew runClient"\n')
        with self.assertRaisesRegex(RuntimeError, "exactly one effective.*cached"):
            self.instance._launch_minecraft(1, self.minecraft)

    def test_ambiguous_effective_commands_are_rejected(self):
        self.script.write_text(SCRIPT+'cmd="./gradlew runClient other"\n')
        runtime.configure_minedojo_runtime(True)
        with self.assertRaisesRegex(RuntimeError, "exactly one effective"):
            self.instance._launch_minecraft(1, self.minecraft)
        self.assertNotIn("--offline", self.script.read_text())

    def test_crlf_and_single_quotes_preserve_original_bytes(self):
        original = b"#!/bin/bash\r\ncmd='./gradlew runClient --stacktrace'\r\n"
        self.script.write_bytes(original)
        runtime.configure_minedojo_runtime(True)
        self.instance._launch_minecraft(1, self.minecraft)
        self.assertEqual(self.script.read_bytes(), original.replace(b"./gradlew ", b"./gradlew --offline "))

    def test_modified_own_copy_is_not_silently_overwritten_when_disabling(self):
        runtime.configure_minedojo_runtime(True)
        self.instance._launch_minecraft(1, self.minecraft)
        changed = self.script.read_text()+"# another component changed this file\n"
        self.script.write_text(changed)
        runtime.configure_minedojo_runtime(False)
        with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
            self.instance._launch_minecraft(2, self.minecraft)
        self.assertEqual(self.script.read_text(), changed)


if __name__ == "__main__":
    unittest.main()
