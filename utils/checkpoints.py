import os
import re
import shutil
import subprocess
import json
from utils.utils import Dprint

def buildObjects(appName):

    # save the current working directory
    cwd = os.getcwd()

    # go to eval-apps directory
    os.chdir("eval-apps")

    # ---- Force-clean the per-app CMake build directory ----
    # Without this, CMake reuses stale object files from the previous
    # BayesOpt iteration (or a previous pipeline run), causing the
    # cross-compiler to silently produce the old hex — the root cause
    # of the "zero cycle reduction" bug.
    app_build_dir = os.path.join("build", appName)
    if os.path.exists(app_build_dir):
        shutil.rmtree(app_build_dir)
        Dprint(f"[buildObjects] Cleaned stale build/{appName}/")

    # check if "build" directory exists
    if not os.path.exists("build"):
        os.makedirs("build")

    # change the directory to the "build" directory
    os.chdir("build")

    # build the appName using cmake
    try:
        result = subprocess.run(
            f"cmake .. -DTARGET_ARCH=msp430 -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && make {appName}-MS-msp430",
            shell=True,
            capture_output=True,
            text=True,
            timeout=float(os.getenv("CHAI_TARGET_BUILD_TIMEOUT", "900")),
        )
        if result.returncode != 0:
            # Surface cross-compiler failures instead of only tracing them.
            # A swallowed failure left a stale hex in place and the campaign
            # then measured the previous candidate's power cycles.
            print(
                f"[buildObjects] MSP430 build FAILED for {appName} "
                f"(exit {result.returncode})"
            )
            Dprint(f"[buildObjects] stderr: {result.stderr[-2000:] if result.stderr else '(empty)'}")
            Dprint(f"[buildObjects] stdout: {result.stdout[-2000:] if result.stdout else '(empty)'}")
            overflow = re.search(
                r"region\s+[`'\"]?(\w+)[`'\"]?\s+overflowed by (\d+) bytes",
                f"{result.stdout}\n{result.stderr}",
                re.IGNORECASE,
            )
            if overflow:
                print(
                    f"[buildObjects] region {overflow.group(1)} overflowed by "
                    f"{overflow.group(2)} bytes — candidate does not fit the device"
                )
    except subprocess.TimeoutExpired:
        Dprint(f"[buildObjects] MSP430 build for {appName} timed out")
    except subprocess.CalledProcessError as e:
        Dprint(f"An error occurred while building the objects: {e}")

    # change back to the original working directory
    os.chdir(cwd)

    # return the path to the built hex file
    hex_path = f"eval-apps/build/{appName}/{appName}-MS-msp430.hex"

    # ---- Verify the hex file actually exists and is fresh ----
    if not os.path.exists(hex_path):
        Dprint(f"[buildObjects] CRITICAL: hex file was NOT produced: {hex_path}")
        Dprint(f"[buildObjects] MSP430_GCC_ROOT={os.environ.get('MSP430_GCC_ROOT', 'NOT SET')}")
        Dprint(f"[buildObjects] MSP430_INC={os.environ.get('MSP430_INC', 'NOT SET')}")
    else:
        import time as _t
        age = _t.time() - os.path.getmtime(hex_path)
        if age > 60:  # older than 60 seconds → probably stale
            Dprint(f"[buildObjects] WARNING: hex file is {age:.0f}s old — may be stale")
        else:
            Dprint(f"[buildObjects] hex file OK: {hex_path} (age {age:.1f}s)")

    return hex_path

def runFused(pathToHex: str):
    # copy the hex file to the fused directory, and rename it to "app.hex"

    binaryName = "fusedBin/app.hex"

    if not os.path.exists(pathToHex):
        Dprint(f"[runFused] ERROR: hex file does not exist: {pathToHex} (cwd={os.getcwd()})")
        return

    # ---- Guard: verify config.yaml exists ----
    config_path = "fusedBin/config.yaml"
    if not os.path.exists(config_path):
        Dprint(f"[runFused] ERROR: Fused config missing: {config_path}")
        return

    # ---- Remove stale cycles.dump BEFORE running ----
    # If Fused crashes or produces no output, we don't want getCyclesFromFused()
    # to silently read a cycles.dump from a previous iteration.
    dump_file = "fusedBin/cycles.dump"
    if os.path.exists(dump_file):
        os.remove(dump_file)
        Dprint(f"[runFused] Removed stale {dump_file}")

    shutil.copy(pathToHex, binaryName)
    Dprint(f"[runFused] Copied {pathToHex} -> {binaryName}, exists={os.path.exists(binaryName)}")
    cwd = os.getcwd()

    # change the directory to the fused directory
    os.chdir("fusedBin")

    # run the fused simulator, wait for it to finish
    #
    # Fused previously ran with no timeout. A simulator that hangs on a
    # pathological candidate blocked the entire campaign indefinitely with no
    # diagnostic, which is indistinguishable from a very slow trace.
    timeout_seconds = float(os.getenv("CHAI_FUSED_TIMEOUT", "1800"))
    try:
        result = subprocess.run(
            "./fused",
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        Dprint(
            f"[runFused] Fused exceeded {timeout_seconds}s and was killed; "
            "treating this candidate as a simulation failure"
        )
        os.chdir(cwd)
        if os.path.exists(binaryName):
            os.remove(binaryName)
        return
    if result.returncode != 0:
        Dprint(f"[runFused] Fused exited with code {result.returncode}")
        if result.stderr:
            Dprint(f"[runFused] stderr (last 500 chars): {result.stderr[-500:]}")

    # change back to the original working directory
    os.chdir(cwd)

    # remove the copied hex file
    if os.path.exists(binaryName):
        os.remove(binaryName)
    else:
        Dprint(f"[runFused] WARNING: {binaryName} not found for cleanup")

def getCyclesFromFused():
    
    dump_file = "fusedBin/cycles.dump"

    if not os.path.exists(dump_file):
        Dprint(f"[getCyclesFromFused] ERROR: {dump_file} not found — Fused may have crashed")
        return None

    # read the cycles from the dump file
    with open(dump_file, "r") as f:
        cycles = f.read().strip()

    if not cycles:
        Dprint(f"[getCyclesFromFused] ERROR: {dump_file} is empty")
        os.remove(dump_file)
        return None

    # subprocess.run(f"rm -f {dump_file}")
    os.remove(dump_file)

    try:
        return int(cycles)
    except ValueError:
        Dprint(f"[getCyclesFromFused] ERROR: non-numeric content in cycles.dump: '{cycles}'")
        return None


def copyNonMakefiles(targetDir, appName):
    # Define the destination directory path
    destinationDir = f"eval-apps/{appName}/"
    
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destinationDir):
        os.makedirs(destinationDir)
    
    # Loop through all the files in the targetDir
    for root, dirs, files in os.walk(targetDir):
        for file in files:
            # Ignore files that are makefiles (case-insensitive)
            if file.lower() == "makefile":
                continue
            
            # Construct full file path
            source_file = os.path.join(root, file)
            destination_file = os.path.join(destinationDir, file)
            
            # Copy file from source to destination
            shutil.copy2(source_file, destination_file)
            # Dprint(f"Copied {source_file} to {destination_file}")



def checkpointOrchestration(targetDir, appName): # Fused intigration need.
    """
    Orchestrates the checkpointing process.

    Args:
        targetDir (str): The directory containing the source to compile.
        appName (str): The name of the application.

    Returns:
        int or None: Number of cycles, or None if the build/simulation failed.
    """

    # Copy files from target/ (or knob_tuning/) to eval-apps/{appName}/
    copyNonMakefiles(targetDir, appName)

    hexDir = buildObjects(appName)

    # Guard: if hex file doesn't exist, abort early instead of crashing Fused
    if not os.path.exists(hexDir):
        Dprint(f"[checkpointOrchestration] ABORT: hex file missing: {hexDir}")
        # Restore eval-apps source to pristine state
        _restore_eval_app_source(appName)
        return None

    runFused(hexDir)

    cycles = getCyclesFromFused()

    # Restore eval-apps/{appName}/ to pristine benchmark state so the next
    # BayesOpt iteration (or next pipeline run) doesn't inherit stale files.
    _restore_eval_app_source(appName)

    if cycles is None:
        Dprint(f"[checkpointOrchestration] WARNING: Fused returned no cycles")
        return None

    return cycles


def _restore_eval_app_source(appName):
    """Restore eval-apps/{appName}/ from benchmark_applications/ to undo
    any modifications made by copyNonMakefiles during this iteration."""
    bm_src = f"benchmark_applications/{appName}/"
    ea_dst = f"eval-apps/{appName}/"
    if os.path.isdir(bm_src):
        if os.path.exists(ea_dst):
            shutil.rmtree(ea_dst)
        shutil.copytree(bm_src, ea_dst)
    else:
        Dprint(f"[_restore_eval_app_source] WARNING: {bm_src} does not exist — cannot restore")
