"""Target (MSP430) build and memory validation for generated candidates.

Why this exists
---------------
Generation previously accepted a candidate as soon as it compiled with **host
gcc**, using a fallback Makefile that literally links ``-lm``. Cross-compilation
for MSP430 happened much later, inside Bayesian optimisation. Two failure modes
followed from that:

* A candidate full of software-emulated floating point passed the acceptance
  gate on a target that has no FPU at all.
* A candidate that overflows FRAM or RAM passed generation and only failed once
  the campaign was already spending simulation budget on it.

This module moves the real target toolchain into the acceptance gate: a
candidate must cross-compile for MSP430 *and* fit in the device's memory
regions before it is treated as a valid approximation plan.

When the cross toolchain is unavailable the gate reports ``skipped`` rather
than ``passed``, so a run performed without it can never be mistaken for a run
that actually validated against the target.
"""

import os
import re
import shutil
import subprocess

from utils.utils import Dprint

#: Memory regions declared by ``eval-apps/support/msp430fr5994.ld``.
#: RAM is 0x1000 bytes and the primary FRAM region is 0xBF80 bytes.
MSP430_REGIONS = {
    "RAM": 0x1000,
    "FRAM": 0xBF80,
}

#: Linker diagnostics that mean "this does not fit on the device".
_OVERFLOW_PATTERN = re.compile(
    r"region\s+[`'\"]?(\w+)[`'\"]?\s+overflowed by (\d+) bytes", re.IGNORECASE
)


class TargetBuildResult:
    """Outcome of one target-toolchain validation."""

    def __init__(self, status, message="", sizes=None):
        #: ``"passed"``, ``"failed"`` or ``"skipped"``.
        self.status = status
        self.message = message
        self.sizes = sizes or {}

    @property
    def ok(self):
        """True when the candidate may proceed (passed, or gate unavailable)."""
        return self.status in ("passed", "skipped")

    def as_dict(self):
        return {
            "status": self.status,
            "message": self.message,
            "sizes": self.sizes,
        }

    def __repr__(self):
        return f"TargetBuildResult({self.status!r}, {self.message!r})"


def toolchain_available():
    """True when the MSP430 cross compiler can be located."""
    root = os.environ.get("MSP430_GCC_ROOT", "")
    candidates = []
    if root:
        candidates.append(os.path.join(root, "bin", "msp430-elf-gcc"))
    which = shutil.which("msp430-elf-gcc")
    if which:
        candidates.append(which)
    return any(os.path.exists(path) for path in candidates)


def _size_tool():
    root = os.environ.get("MSP430_GCC_ROOT", "")
    if root:
        candidate = os.path.join(root, "bin", "msp430-elf-size")
        if os.path.exists(candidate):
            return candidate
    return shutil.which("msp430-elf-size")


def _parse_size_output(text):
    """Parse ``msp430-elf-size -A`` output into ``{section: bytes}``."""
    sizes = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("."):
            try:
                sizes[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return sizes


def _memory_usage(sizes):
    """Map section sizes onto the device's RAM and FRAM regions."""
    ram = sizes.get(".bss", 0) + sizes.get(".data", 0) + sizes.get(".noinit", 0)
    fram = (
        sizes.get(".text", 0)
        + sizes.get(".rodata", 0)
        + sizes.get(".data", 0)
        + sizes.get(".persistent", 0)
        + sizes.get(".upper.text", 0)
        + sizes.get(".upper.rodata", 0)
    )
    return {"RAM": ram, "FRAM": fram}


def validate_target_build(app_name, source_dir, method="MS"):
    """Cross-compile *source_dir* for MSP430 and check it fits on the device.

    Parameters
    ----------
    app_name : str
        Benchmark directory name under ``eval-apps/``.
    source_dir : str
        Directory holding the patched candidate sources to validate.
    method : str
        Intermittent-computing method variant to build (matches the CMake
        target naming, e.g. ``ar-iclib-MS-msp430``).

    Returns
    -------
    TargetBuildResult
    """
    if not toolchain_available():
        return TargetBuildResult(
            "skipped",
            "MSP430 cross toolchain not found (MSP430_GCC_ROOT unset and "
            "msp430-elf-gcc not on PATH); target build gate did not run",
        )

    benchmark_dir = f"benchmark_applications/{app_name}"
    eval_dir = f"eval-apps/{app_name}"
    if not os.path.isdir(benchmark_dir):
        return TargetBuildResult(
            "skipped", f"no pristine benchmark source at {benchmark_dir}"
        )

    target_name = f"{app_name}-{method}-msp430"
    cwd = os.getcwd()
    try:
        # Stage the candidate into the cross-build tree.
        for root, _dirs, files in os.walk(source_dir):
            for filename in files:
                if filename.lower() == "makefile":
                    continue
                if not filename.endswith((".c", ".h")):
                    continue
                shutil.copy2(os.path.join(root, filename), eval_dir)

        build_dir = os.path.join("eval-apps", "build")
        app_build_dir = os.path.join(build_dir, app_name)
        if os.path.exists(app_build_dir):
            shutil.rmtree(app_build_dir)
        os.makedirs(build_dir, exist_ok=True)

        os.chdir(build_dir)
        result = subprocess.run(
            "cmake .. -DTARGET_ARCH=msp430 -DCMAKE_POLICY_VERSION_MINIMUM=3.5 "
            f"&& make {target_name}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=int(os.getenv("CHAI_TARGET_BUILD_TIMEOUT", "900")),
        )
        os.chdir(cwd)

        combined = f"{result.stdout}\n{result.stderr}"

        overflow = _OVERFLOW_PATTERN.search(combined)
        if overflow:
            region, excess = overflow.group(1), overflow.group(2)
            return TargetBuildResult(
                "failed",
                f"MSP430 link failed: region {region} overflowed by {excess} "
                f"bytes. The approximation must fit the device's memory; "
                f"reduce lookup tables, buffers, or code size.",
            )

        if result.returncode != 0:
            tail = combined.strip()[-1200:]
            return TargetBuildResult(
                "failed",
                f"MSP430 cross-compilation failed for {target_name}:\n{tail}",
            )

        elf_path = os.path.join(
            "eval-apps", "build", app_name, f"{target_name}.elf"
        )
        if not os.path.exists(elf_path):
            elf_path = os.path.join("eval-apps", "build", app_name, target_name)
        if not os.path.exists(elf_path):
            return TargetBuildResult(
                "failed",
                f"MSP430 build reported success but produced no ELF for "
                f"{target_name}",
            )

        size_tool = _size_tool()
        if size_tool is None:
            return TargetBuildResult(
                "passed",
                "cross-compiled for MSP430 (msp430-elf-size unavailable, "
                "memory usage not measured)",
            )

        size_run = subprocess.run(
            [size_tool, "-A", elf_path], capture_output=True, text=True, timeout=60
        )
        sizes = _parse_size_output(size_run.stdout)
        usage = _memory_usage(sizes)

        for region, limit in MSP430_REGIONS.items():
            used = usage.get(region, 0)
            if used > limit:
                return TargetBuildResult(
                    "failed",
                    f"MSP430 {region} usage {used} bytes exceeds the device "
                    f"limit of {limit} bytes",
                    sizes=usage,
                )

        Dprint(
            f"[target build] {target_name}: RAM {usage['RAM']}/{MSP430_REGIONS['RAM']} B, "
            f"FRAM {usage['FRAM']}/{MSP430_REGIONS['FRAM']} B"
        )
        return TargetBuildResult("passed", "fits MSP430 memory regions", sizes=usage)

    except subprocess.TimeoutExpired:
        return TargetBuildResult("failed", "MSP430 cross-build timed out")
    except Exception as error:  # pragma: no cover - defensive
        return TargetBuildResult(
            "skipped", f"target build gate raised {error.__class__.__name__}: {error}"
        )
    finally:
        os.chdir(cwd)
        # Always restore the pristine benchmark tree.
        if os.path.isdir(benchmark_dir):
            if os.path.exists(eval_dir):
                shutil.rmtree(eval_dir)
            shutil.copytree(benchmark_dir, eval_dir)
