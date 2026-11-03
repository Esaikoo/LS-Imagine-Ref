"""Optional process-local MineDojo launch adjustments on managed temp copies.

Nothing here edits the installed MineDojo tree or Gradle's cache/settings. The
default is unchanged online Gradle resolution. Offline mode needs this machine's
already cached build dependencies and is useful for reproducible local checks.
"""
import functools
import logging
import os
from pathlib import Path
import re
import shlex
import stat
import tempfile
import threading

_enabled = False
_lock = threading.RLock()
_wrapped_classes = {}
_edits = {}
_CACHE_HINT = "MineDojo offline Gradle mode requires build dependencies already cached on this machine."
_COMMAND = re.compile(r'''(?m)^(?P<prefix>[ \t]*cmd[ \t]*=[ \t]*(?P<quote>["'])\./gradlew)(?P<arguments>[^\r\n]*)(?P=quote)[ \t]*\r?$''')


def _load_minecraft_runtime():
    from minedojo.sim.bridge.mc_instance.instance import MinecraftInstance, MC_DIR
    return MinecraftInstance, Path(MC_DIR)


def _failure(message):
    return RuntimeError(message+" "+_CACHE_HINT)


def _managed_script(instance, minecraft_dir, source_dir):
    try:
        raw_root = getattr(instance, "instance_dir", None)
        if not raw_root:
            raise ValueError("instance_dir is missing")
        root = Path(raw_root).resolve(strict=True)
        minecraft = Path(minecraft_dir).resolve(strict=True)
        expected = (root/"Minecraft").resolve(strict=True)
        source = Path(source_dir).resolve()
        if not root.is_dir() or not minecraft.is_dir() or minecraft != expected:
            raise ValueError("minecraft_dir is not this instance's managed Minecraft copy")
        if minecraft.parent != root or minecraft.name != "Minecraft":
            raise ValueError("managed Minecraft directory escapes instance_dir")
        if minecraft == source or source in minecraft.parents:
            raise ValueError("refusing to edit the installed MineDojo source tree")
        script = minecraft/"launchClient.sh"
        if script.is_symlink() or not script.is_file() or script.resolve().parent != minecraft:
            raise ValueError("launchClient.sh must be a regular file inside the managed copy")
        if script.stat().st_nlink != 1:
            raise ValueError("refusing to edit a shared/hardlinked launch script")
        return script
    except (OSError, TypeError, ValueError) as error:
        raise _failure("Unsafe or unavailable managed Minecraft launch path: "+str(error)) from error


def _replace_bytes(script, value):
    """Publish only this checked copy, retaining its executable permission bits."""
    mode = stat.S_IMODE(script.stat().st_mode)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".launchClient-offline-", dir=script.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(value)
        temporary.chmod(mode)
        os.replace(temporary, script)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _enable_on_copy(instance, minecraft_dir, source_dir):
    script = _managed_script(instance, minecraft_dir, source_dir)
    original = script.read_bytes()
    try:
        text = original.decode("utf-8")
        matches = [match for match in _COMMAND.finditer(text)
                   if "runClient" in shlex.split(match.group("arguments"))]
    except (UnicodeError, ValueError) as error:
        raise _failure("Cannot parse the managed Minecraft Gradle launch command.") from error
    if len(matches) != 1:
        raise _failure("Expected exactly one effective Gradle runClient command in the managed launch script.")
    match = matches[0]
    if "--offline" in shlex.split(match.group("arguments")):
        return
    changed = (text[:match.end("prefix")]+" --offline"+text[match.end("prefix"):]).encode("utf-8")
    _replace_bytes(script, changed)
    _edits[str(script)] = (original, changed)


def _restore_own_copy(instance, minecraft_dir, source_dir):
    # Default/online calls do no filesystem work unless this exact temporary
    # script was previously changed by this module in the same process.
    candidate = str((Path(minecraft_dir)/"launchClient.sh").resolve())
    record = _edits.get(candidate)
    if record is None:
        return
    if not Path(candidate).exists():
        _edits.pop(candidate, None)
        return
    script = _managed_script(instance, minecraft_dir, source_dir)
    original, changed = record
    if script.read_bytes() != changed:
        raise _failure("Managed launch script changed since offline configuration; refusing to overwrite it.")
    _replace_bytes(script, original)
    _edits.pop(candidate, None)


def configure_minedojo_runtime(offline=False):
    """Configure future launches in this process; False keeps the default flow.

    Only the managed copy at self.instance_dir/Minecraft may be changed, directly
    before the original launcher runs. Switching off restores our own edit if a
    temporary copy is relaunched; it never removes an existing user's --offline.
    Already running processes and their launch arguments are unaffected.
    """
    global _enabled
    with _lock:
        if not offline:
            _enabled = False
            return
        instance_class, source_dir = _load_minecraft_runtime()
        if instance_class not in _wrapped_classes:
            original = instance_class._launch_minecraft

            @functools.wraps(original)
            def launch(instance, port, minecraft_dir, replaceable=True, *args, **kwargs):
                with _lock:
                    enabled = _enabled
                    if enabled:
                        _enable_on_copy(instance, minecraft_dir, source_dir)
                    else:
                        _restore_own_copy(instance, minecraft_dir, source_dir)
                try:
                    return original(instance, port, minecraft_dir, replaceable, *args, **kwargs)
                except Exception as error:
                    if enabled:
                        raise _failure("Managed Minecraft launch failed: "+str(error)) from error
                    raise

            instance_class._launch_minecraft = launch
            _wrapped_classes[instance_class] = original
        _enabled = True
        logging.getLogger(__name__).info("Offline Gradle launch enabled for managed temporary copies only. %s", _CACHE_HINT)
