"""Verify that a candidate's declared identity setting really is the identity.

Why this exists
---------------
The generation prompt requires every approximation to expose a knob setting at
which the function reproduces the original behaviour exactly, and BayesOpt
anchors its whole search on that point. But the check lived in
``lib/bo.py``, which runs *after* the entire generation loop has finished. By
then the only available response to a broken identity point is to discard the
plan for the whole function -- the model never learns it got this wrong, and
never gets a chance to fix it.

Moving the check into the generation acceptance gate turns a silent plan
discard into an ordinary, repairable compile-style rejection: the candidate is
built at its declared defaults, run, and scored against the reference output.
Anything above the identity tolerance is handed back to the model with a
specific message.

This is deliberately the same measurement BayesOpt makes later, so a candidate
that passes here cannot fail the identity gate there.
"""

import os
import shutil

from utils.error_analyzer import evaluate_error, ground_truth_files
from utils.optimization_metrics import IDENTITY_TOLERANCE, is_identity_error
from utils.utils import Dprint


class IdentityCheckUnavailable(RuntimeError):
    """Raised when the check cannot run, as distinct from failing."""


def _stage_reference(app_name, candidate_dir, reference_dir="target"):
    """Copy the reference outputs the evaluator needs into the candidate dir."""
    artifacts = ground_truth_files(app_name)
    if not artifacts:
        raise IdentityCheckUnavailable(
            f"no reference artifact registered for {app_name}"
        )
    for name in artifacts:
        source = os.path.join(reference_dir, name)
        if not os.path.exists(source):
            raise IdentityCheckUnavailable(
                f"reference output {source} has not been generated yet"
            )
        shutil.copy2(source, os.path.join(candidate_dir, name))


def verify_identity(app_name, candidate_dir, reference_dir="target"):
    """Return ``(ok, message)`` for the candidate at its declared defaults.

    ``ok`` is True when the candidate reproduces the reference output. The
    candidate directory is expected to already contain the patched source built
    with the knobs at their declared initial values, which are the identity
    values by contract.
    """
    if not os.path.isdir(candidate_dir):
        raise IdentityCheckUnavailable(f"no candidate directory {candidate_dir}")

    _stage_reference(app_name, candidate_dir, reference_dir)

    try:
        error = evaluate_error(app_name, candidate_dir)
    except Exception as failure:
        # A build or run failure here is a real defect in the candidate, not an
        # unavailable check: the same code is about to be handed to BayesOpt.
        return False, (
            "The approximated function could not be built and run at its "
            f"declared identity knob values: {failure}"
        )

    if is_identity_error(error):
        Dprint(f"[identity gate] {app_name}: identity verified (error {error})")
        return True, ""

    return False, (
        f"The declared identity knob values do not reproduce the original "
        f"function: measured error {error:.6f}, which exceeds the identity "
        f"tolerance {IDENTITY_TOLERANCE}. Every knob's initial value must be "
        "the setting at which this function behaves EXACTLY as the original "
        "(for example a scale or stride of 1, an enable flag of 0, or an "
        "iteration budget at its upper bound). Fix the initial values, or the "
        "guard conditions that are supposed to bypass the approximation at "
        "those values, so that the unapproximated path runs by default."
    )
