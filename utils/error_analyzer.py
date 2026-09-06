import os
import numpy as np
import cv2
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import subprocess
from utils.utils import Dprint


#: What each benchmark's reported "error" actually measures, and the bound that
#: applies to it.
#:
#: Normalising six different losses into [0, 1] makes them *arithmetically*
#: comparable. It does NOT make them semantically equivalent: 0.30 of SSIM loss,
#: 0.30 of F1 loss, and 0.30 of normalised spectral deviation are different
#: amounts of damage to different kinds of output. Each benchmark therefore
#: carries its own bound and a stated justification for it, rather than
#: inheriting a single global 0.30 that was never argued for per task.
#:
#: Override any of these with CHAI_ERROR_BOUND_<APP> (dashes as underscores,
#: upper case), e.g. CHAI_ERROR_BOUND_SOBEL_ICLIB=0.2.
ERROR_METRIC_SPEC = {
    "sobel-iclib": {
        "metric": "1 - SSIM between output and reference edge images",
        "bound": 0.30,
        "justification": (
            "SSIM loss above ~0.3 visibly destroys edge structure; below it the "
            "edge map remains usable for downstream detection."
        ),
    },
    "accept-sobel": {
        "metric": "1 - SSIM between output and reference edge images",
        "bound": 0.30,
        "justification": "Same output type and reasoning as sobel-iclib.",
    },
    "susan": {
        "metric": "1 - F1 of detected corners against the reference",
        "bound": 0.30,
        "justification": (
            "F1 below 0.70 means roughly a third of corner decisions are wrong, "
            "which breaks correspondence for any consumer of the corner set."
        ),
    },
    "stringsearch-iclib": {
        "metric": "1 - F1 of per-query match/no-match decisions",
        "bound": 0.20,
        "justification": (
            "Tighter than the others: a search result is a discrete decision "
            "with no graceful degradation. One wrong answer in five is already "
            "a large behavioural change, unlike a slightly blurrier image."
        ),
    },
    "fft-iclib": {
        "metric": "mean |X_ref - X_approx| normalised by mean |X_ref|",
        "bound": 0.30,
        "justification": (
            "A spectrum within 30% of the reference magnitude preserves peak "
            "ordering, which is what downstream feature extraction uses."
        ),
    },
    "ar-iclib": {
        "metric": "per-window normalised mean absolute error of class counts",
        "bound": 0.30,
        "justification": (
            "Activity recognition tolerates per-window count error while the "
            "dominant class per window is preserved."
        ),
    },
    "accept-activityrec": {
        "metric": "per-window normalised mean absolute error of class counts",
        "bound": 0.30,
        "justification": "Same output type and reasoning as ar-iclib.",
    },
    "lqi-iclib": {
        "metric": "relative absolute error of the link estimate (zero-safe)",
        "bound": 0.30,
        "justification": (
            "Link-quality estimates drive coarse routing decisions and are "
            "already noisy; 30% relative error preserves the ordering of links."
        ),
    },
    "link-estimator": {
        "metric": "relative absolute error of the link estimate (zero-safe)",
        "bound": 0.30,
        "justification": "Same output type and reasoning as lqi-iclib.",
    },
    "bc-iclib": {
        "metric": "elementwise normalised absolute error of bit counts",
        "bound": 0.20,
        "justification": (
            "Tighter than the others: a bit count is an exact integer answer "
            "with a small dynamic range, so a given normalised error is a much "
            "larger semantic error than the same number on a continuous signal."
        ),
    },
    "radix-bm": {
        "metric": "elementwise normalised absolute error of sorted output",
        "bound": 0.30,
        "justification": "Ordering-sensitive numeric output; matches the suite default.",
    },
    "segment-bm": {
        "metric": "elementwise normalised absolute error of segment output",
        "bound": 0.30,
        "justification": "Matches the suite default for continuous outputs.",
    },
}


def error_bound_for(app_name, default=None):
    """Return the accuracy bound that applies to *app_name*.

    A single suite-wide 0.30 treated an image-similarity loss and an exact
    integer-answer loss as equivalent. Each benchmark's bound is declared above
    with its justification, and can be overridden per benchmark from the
    environment for a sensitivity analysis.
    """
    override = os.getenv(
        "CHAI_ERROR_BOUND_" + app_name.replace("-", "_").upper()
    )
    if override:
        return float(override)
    spec = ERROR_METRIC_SPEC.get(app_name)
    if spec:
        return float(spec["bound"])
    if default is not None:
        return float(default)
    from config.config import ERROR_BOUND

    return float(ERROR_BOUND)


def describe_error_metric(app_name):
    """Return the human-readable definition of *app_name*'s error metric."""
    spec = ERROR_METRIC_SPEC.get(app_name)
    return spec["metric"] if spec else "unspecified"


def describe_error_bound(app_name):
    """Return ``(bound, justification)`` for *app_name*."""
    spec = ERROR_METRIC_SPEC.get(app_name)
    if not spec:
        return error_bound_for(app_name), "no benchmark-specific justification"
    return error_bound_for(app_name), spec["justification"]


def _normalized(value):
    """Clamp a normalised error into [0, 1] and reject non-finite results."""
    value = float(value)
    if not np.isfinite(value):
        raise RuntimeError(f"error evaluation produced a non-finite value: {value}")
    return float(min(max(value, 0.0), 1.0))


def _build_checked(app_label):
    """Clean-build the candidate in the current directory, failing loudly."""
    clean = subprocess.run(
        "make clean", shell=True, capture_output=True, text=True
    )
    if clean.returncode != 0:
        raise RuntimeError(
            f"{app_label} clean failed: {clean.stdout}{clean.stderr}".strip()
        )
    build = subprocess.run("make main", shell=True, capture_output=True, text=True)
    if build.returncode != 0:
        raise RuntimeError(
            f"{app_label} build failed: {build.stdout}{build.stderr}".strip()
        )


def _clear_stale(*paths):
    """Delete outputs from a previous iteration before re-running.

    Without this a failed run leaves the previous candidate's output in place
    and the next error measurement silently scores stale data.
    """
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def _run_main_to_file(output_path, *, timeout=30, tolerate_nonzero=False):
    """Run the benchmark without a shell and bound its execution time."""
    with open(output_path, "w", encoding="utf-8") as output:
        run = subprocess.run(
            ["./main"],
            stdout=output,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    if run.returncode != 0 and not tolerate_nonzero:
        raise RuntimeError(
            f"benchmark execution failed with exit {run.returncode}: "
            f"{run.stderr.strip()}"
        )
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"benchmark execution produced no {output_path}")
    return run


def generateGroundTruth(appName):
    """Build and run the unapproximated program to produce the reference output.

    Every branch used to run ``make clean; make main; ./main > out`` with no
    status check anywhere. A failed reference build therefore left the previous
    benchmark's ground-truth file in place, and every error measurement in the
    campaign was silently taken against the wrong reference. Each step is now
    checked, and the stale artifact is deleted before the run so a failure can
    never masquerade as a success.
    """
    _IMAGE_APPS = {"susan", "sobel-iclib", "sobel-esp", "accept-sobel"}
    _CSV_APPS = {
        "stringsearch", "stringsearch-iclib", "ar-iclib", "fft-iclib",
        "bc-iclib", "radix-bm", "segment-bm", "accept-activityrec",
    }
    _TXT_APPS = {"fft", "lqi", "lqi-iclib", "link-estimator"}

    if appName in _IMAGE_APPS:
        artifact, intermediate = "out.pgm", "img.pgm"
    elif appName in _CSV_APPS:
        artifact, intermediate = "ground_truth.csv", None
    elif appName in _TXT_APPS:
        artifact, intermediate = "ground_truth.txt", None
    else:
        raise ValueError(f"Invalid application name: {appName}")

    cwd = os.getcwd()
    try:
        os.chdir("target")

        _clear_stale(artifact, *( [intermediate] if intermediate else [] ))
        _build_checked(f"{appName} (ground truth)")

        if appName == "susan":
            with open(artifact, "w") as output:
                run = subprocess.run(
                    ["./main"], stdout=output, stderr=subprocess.PIPE,
                    text=True, timeout=120,
                )
        elif intermediate:
            run = subprocess.run(
                ["./main"], capture_output=True, text=True, timeout=120
            )
        else:
            with open(artifact, "w") as output:
                run = subprocess.run(
                    ["./main"], stdout=output, stderr=subprocess.PIPE,
                    text=True, timeout=120,
                )

        # bc-iclib declares `void main()` and can return a garbage status even
        # on success; a signal is still a real crash.
        crashed = run.returncode < 0 or run.returncode >= 128
        if crashed or (run.returncode != 0 and appName != "bc-iclib"):
            raise RuntimeError(
                f"ground-truth run for {appName} failed with exit "
                f"{run.returncode}: {(run.stderr or '').strip()}"
            )

        if intermediate and appName != "susan":
            if not os.path.exists(intermediate):
                raise RuntimeError(
                    f"ground-truth run for {appName} produced no {intermediate}"
                )
            os.rename(intermediate, artifact)

        if not os.path.exists(artifact) or os.path.getsize(artifact) == 0:
            raise RuntimeError(
                f"ground-truth run for {appName} produced no usable {artifact}"
            )

        Dprint(
            f"[ground truth] {appName}: wrote {artifact} "
            f"({os.path.getsize(artifact)} bytes)"
        )
    finally:
        os.chdir(cwd)


def load_pgm_opencv(file_path):
    # Load PGM image using OpenCV
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    return image

def sobelError(pathToCodebase):
    """Return ``1 - SSIM`` between the reference and approximate edge images.

    The previous implementation logged ``1 - SSIM`` but returned
    ``(1 - SSIM) / 2``, so every reported Sobel error was half the damage
    actually measured and the shared 30% bound was really a 60% bound on this
    benchmark alone. It also ran ``./main && mv img.pgm predicted.pgm`` without
    checking the exit status, so a candidate that failed to run was scored
    against the *previous* candidate's image.
    """
    cwd = os.getcwd()
    try:
        os.chdir(pathToCodebase)
        _build_checked("sobel-iclib")

        predicted = "predicted.pgm"
        _clear_stale(predicted, "img.pgm")

        run = subprocess.run(["./main"], capture_output=True, text=True, timeout=60)
        if run.returncode != 0:
            raise RuntimeError(
                f"sobel-iclib execution failed with exit {run.returncode}: "
                f"{run.stderr.strip()}"
            )
        if not os.path.exists("img.pgm"):
            raise RuntimeError("sobel-iclib produced no img.pgm")
        os.rename("img.pgm", predicted)

        original_image = load_pgm_opencv("out.pgm")
        approximated_image = load_pgm_opencv(predicted)
        if original_image is None or approximated_image is None:
            raise RuntimeError("sobel-iclib reference or output image is unreadable")
        if original_image.shape != approximated_image.shape:
            raise RuntimeError(
                f"sobel-iclib output shape {approximated_image.shape} does not "
                f"match the reference {original_image.shape}"
            )

        ssim_score = ssim(original_image, approximated_image)
        error = _normalized(1 - ssim_score)
        Dprint(f"Debug: SSIM: {ssim_score}  Error (1 - SSIM): {error}")
        return error
    finally:
        os.chdir(cwd)


def susanError(pathToCodebase):
    _entry_cwd = os.getcwd()
    try:

        # Save the current working directory
        cwd = os.getcwd()

        # Change directory to the codebase
        os.chdir(pathToCodebase)

        ground_truth = "out.pgm"
        predicted = "predicted.pgm"

        # Make clean and make main
        subprocess.run("make clean", shell=True, stdout=subprocess.DEVNULL)
        subprocess.run("make main", shell=True, stdout=subprocess.DEVNULL)

        cmd = f"./main > {predicted}"
        subprocess.run(cmd, shell=True)

        # Calculate the F1-score
        original_image = load_pgm_opencv(ground_truth)
        approximated_image = load_pgm_opencv(predicted)

        true_positives = np.sum((original_image == 255) & (approximated_image == 255))
        false_positives = np.sum((original_image == 0) & (approximated_image == 255))
        false_negatives = np.sum((original_image == 255) & (approximated_image == 0))

        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

        # original_image = load_pgm_opencv(ground_truth)
        # approximated_image = load_pgm_opencv(predicted)
        # ssim_score = ssim(original_image, approximated_image)

        # # Change back to the original working directory
        os.chdir(cwd)

        # Dprint(f"Debug: SSIM Score (Accuracy): {ssim_score}")
        # Dprint(f"Debug: Error: {(1 - ssim_score) / 2}")

        Dprint(f"Debug: F1 Score: {f1_score}")
        Dprint(f"Debug: Error: {1 - f1_score}")

        return _normalized(1 - f1_score)


    finally:
        os.chdir(_entry_cwd)

def stringSearchError(pathToCodebase):
    _entry_cwd = os.getcwd()
    try:

        # Save the current working directory
        cwd = os.getcwd()

        # Change directory to the codebase
        os.chdir(pathToCodebase)

        _build_checked("stringsearch-iclib")
        _clear_stale("output.csv")
        _run_main_to_file("output.csv")

        # Load the ground truth
        with open("ground_truth.csv", "r") as file:
            ground_truth = file.read().splitlines()

        # Load the output
        with open("output.csv", "r") as file:
            predictions = file.read().splitlines()

        # Ensure the DataFrames have the same length
        if len(ground_truth) != len(predictions):
            raise ValueError(
                "The number of rows in ground_truth and output must be the same."
            )

        # Calculate the F1-score
        tp = 0
        fp = 0
        fn = 0

        # A prediction that is neither "0" nor "1" used to match none of these
        # branches, so a candidate emitting garbage scored as a perfect run.
        for i in range(len(ground_truth)):
            truth = ground_truth[i].strip()
            prediction = predictions[i].strip()

            if truth not in ("0", "1"):
                raise RuntimeError(
                    f"stringsearch-iclib reference line {i} is {truth!r}, "
                    "expected '0' or '1'"
                )
            if prediction not in ("0", "1"):
                raise RuntimeError(
                    f"stringsearch-iclib produced {prediction!r} on line {i}; "
                    "the benchmark must emit only '0' or '1'"
                )

            if truth == "1" and prediction == "1":
                tp += 1
            elif truth == "0" and prediction == "1":
                fp += 1
            elif truth == "1" and prediction == "0":
                fn += 1

        Dprint(f"TP: {tp}, FP: {fp}, FN: {fn}")

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (fn + tp) if (fn + tp) > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Change back to the original working directory
        os.chdir(cwd)

        Dprint(f"F1 Score: {f1_score}")

        return _normalized(1 - f1_score)

    finally:
        os.chdir(_entry_cwd)

def fftError(pathToCodebase):
    _entry_cwd = os.getcwd()
    try:
        # Save the current working directory
        cwd = os.getcwd()

        # Change directory to the codebase
        os.chdir(pathToCodebase)

        _build_checked("fft-iclib")
        _clear_stale("output.csv")
        _run_main_to_file("output.csv")

        # # Make sure the arrays are of the same length
        # if len(ground_truth_fft) != len(approximate_fft):
        #     # Truncate both to the shorter length
        #     min_length = min(len(ground_truth_fft), len(approximate_fft))
        #     ground_truth_fft = ground_truth_fft[:min_length]
        #     approximate_fft = approximate_fft[:min_length]

        # # Calculate the magnitude of both the accurate and approximate FFT results
        # original_magnitude = np.abs(ground_truth_fft)  # Magnitude of the ground truth
        # approximate_magnitude = np.abs(approximate_fft)  # Magnitude of the approximate FFT

        # # Calculate R2-score using magnitudes
        # residual_sum_of_squares = np.sum((original_magnitude - approximate_magnitude) ** 2)
        # total_sum_of_squares = np.sum((original_magnitude - np.mean(original_magnitude)) ** 2)

        # Dprint(original_magnitude , np.mean(original_magnitude))
        # Dprint(residual_sum_of_squares )

        # r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)

        import csv

        def read_fft_from_csv(filename):
            """
            Reads FFT data from a CSV file where each row contains two values: 
            the real part and the imaginary part.
            """
            fft_results = []
            with open(filename, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    real_part = float(row[0])
                    imag_part = float(row[1])
                    fft_results.append(complex(real_part, imag_part))
            return np.array(fft_results)

        def relative_error(fft_accurate, fft_inaccurate):
            """Spectrum-normalised mean complex error.

            The previous per-bin form divided by ``|X_accurate|`` and then mapped
            the resulting NaN to 0. A reference bin of exactly zero is normal in an
            FFT, so any error in those bins was scored as a *perfect* match. This
            normalises the mean absolute complex deviation by the mean reference
            magnitude, which is defined for the whole spectrum.
            """
            if len(fft_accurate) != len(fft_inaccurate):
                raise ValueError("FFT result arrays must be of the same length.")

            deviation = np.mean(np.abs(fft_accurate - fft_inaccurate))
            reference = np.mean(np.abs(fft_accurate))
            if reference <= 0:
                raise RuntimeError("fft-iclib reference spectrum has zero magnitude")
            return deviation / reference


        # Load the ground truth FFT results (real, imaginary pairs)
        ground_truth_fft = read_fft_from_csv("ground_truth.csv")

        # Load the approximate FFT results (real, imaginary pairs)
        approximate_fft = read_fft_from_csv("output.csv")


        error = _normalized(relative_error(ground_truth_fft, approximate_fft))

        os.chdir(cwd)

        return error


    finally:
        os.chdir(_entry_cwd)

def lqiError(pathToCodebase):
    """Zero-safe relative absolute error of the link-quality estimate.

    Previously this built without checking the exit status and divided by the
    ground truth with no zero guard, so a legitimate estimate of 0 produced a
    ZeroDivisionError or an infinity that propagated into the objective.
    """
    cwd = os.getcwd()
    try:
        os.chdir(pathToCodebase)
        _build_checked("lqi-iclib")
        _clear_stale("output.txt")
        _run_main_to_file("output.txt")

        with open("ground_truth.txt", "r") as file:
            ground_truth = float(file.read())
        with open("output.txt", "r") as file:
            approximate = float(file.read())

        # Normalise by the larger of |ground truth| and 1 so a zero reference
        # yields a bounded absolute deviation instead of an infinity.
        denominator = max(abs(ground_truth), 1.0)
        return _normalized(abs(ground_truth - approximate) / denominator)
    finally:
        os.chdir(cwd)


def arError(pathToCodebase):
    def calculate_mape_from_csv(pred_file_path, gt_file_path):
        """Per-window normalised mean absolute error of the class counts.

        The previous version summed each column over the whole run and compared
        only the two totals. That discards all temporal/order information: an
        output that assigns the wrong class to every window still scores well
        whenever the per-class totals happen to match. Comparing row by row
        keeps the window structure that the benchmark is actually about.
        """
        pred_df = pd.read_csv(pred_file_path, header=None, names=['still', 'moving'])
        ground_truth_df = pd.read_csv(
            gt_file_path, header=None, names=['still', 'moving']
        )

        if len(pred_df) != len(ground_truth_df):
            raise RuntimeError(
                f"ar-iclib produced {len(pred_df)} windows but the reference "
                f"has {len(ground_truth_df)}"
            )

        predicted = pred_df.to_numpy(dtype=float)
        reference = ground_truth_df.to_numpy(dtype=float)
        # Normalise each window by its own reference magnitude, floored at 1 so
        # an all-zero reference window cannot divide by zero.
        denominator = np.maximum(np.abs(reference), 1.0)
        return float(np.mean(np.abs(reference - predicted) / denominator))


    # A numerical error is meaningful only after a successful build and run.
    # Previously failed commands were ignored and a missing/stale output was
    # misreported as an error-bound violation.
    cwd = os.getcwd()
    try:
        os.chdir(pathToCodebase)
        _build_checked("ar-iclib")
        _clear_stale("output.csv")
        _run_main_to_file("output.csv")
        return _normalized(
            calculate_mape_from_csv("output.csv", "ground_truth.csv")
        )
    finally:
        os.chdir(cwd)

def radixbmError(pathToCodebase):
    """Error function for radix-bm benchmark. Uses relative error on single hash output."""
    cwd = os.getcwd()
    os.chdir(pathToCodebase)

    subprocess.run("make clean", shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("make main", shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("./main > output.csv", shell=True)

    os.chdir(cwd)

    # Read ground truth and predicted values
    with open(f"{pathToCodebase}/ground_truth.csv", "r") as f:
        gt_val = float(f.read().strip())
    with open(f"{pathToCodebase}/output.csv", "r") as f:
        pred_val = float(f.read().strip())

    if gt_val == 0:
        error = 0.0 if pred_val == 0 else 1.0
    else:
        error = abs(gt_val - pred_val) / abs(gt_val)

    Dprint(f"Radix-BM Error: {error}")
    return _normalized(error)


def acceptSobelError(pathToCodebase):
    """Error function for ACCEPT sobel benchmark. Uses SSIM like sobel-iclib."""
    return sobelError(pathToCodebase)


def acceptActivityrecError(pathToCodebase):
    """Error function for ACCEPT activity recognition benchmark. Uses MAPE like ar-iclib."""
    return arError(pathToCodebase)


def segmentbmError(pathToCodebase):
    """Error function for segment-bm benchmark. Uses relative error on single hash output."""
    cwd = os.getcwd()
    os.chdir(pathToCodebase)

    subprocess.run("make clean", shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("make main", shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("./main > output.csv", shell=True)

    os.chdir(cwd)

    # Read ground truth and predicted values
    with open(f"{pathToCodebase}/ground_truth.csv", "r") as f:
        gt_val = float(f.read().strip())
    with open(f"{pathToCodebase}/output.csv", "r") as f:
        pred_val = float(f.read().strip())

    if gt_val == 0:
        error = 0.0 if pred_val == 0 else 1.0
    else:
        error = abs(gt_val - pred_val) / abs(gt_val)

    Dprint(f"Segment-BM Error: {error}")
    return _normalized(error)


def bitcountError(pathToCodebase):
    def calculate_mape_from_csv(pred_file_path, gt_file_path):
        """
        Calculate the Mean Absolute Percentage Error (MAPE) from predicted and ground truth CSV files.

        Parameters:
        pred_file_path (str): Path to the CSV file containing the predicted values.
        gt_file_path (str): Path to the CSV file containing the ground truth values.

        Returns:
        float: The overall MAPE value.
        """
        # Read the predicted CSV file into a DataFrame
        pred_df = pd.read_csv(pred_file_path, header=None)
        
        # Read the ground truth CSV file into a DataFrame
        ground_truth_df = pd.read_csv(gt_file_path, header=None)
        
        if pred_df.shape != ground_truth_df.shape:
            raise RuntimeError(
                f"bc-iclib output shape {pred_df.shape} does not match the "
                f"reference {ground_truth_df.shape}"
            )

        # A reference bit count of 0 is legitimate. Dividing by
        # ``ground_truth + 1e-10`` turned any error on those entries into ~1e10,
        # which saturated the mean and made every other entry irrelevant.
        predicted = pred_df.to_numpy(dtype=float)
        reference = ground_truth_df.to_numpy(dtype=float)
        denominator = np.maximum(np.abs(reference), 1.0)
        return float(np.mean(np.abs(reference - predicted) / denominator))

    cwd = os.getcwd()
    try:
        os.chdir(pathToCodebase)
        subprocess.run("make clean", shell=True, check=True, capture_output=True, text=True)
        build = subprocess.run("make main", shell=True, capture_output=True, text=True)
        if build.returncode != 0:
            raise RuntimeError(
                f"bc-iclib build failed: {build.stdout}{build.stderr}".strip()
            )
        run = subprocess.run(
            ["./main"], capture_output=True, text=True, timeout=30,
        )
        # This legacy benchmark declares ``void main()`` and therefore may
        # return a small garbage status even after producing valid output.
        # A signal (negative direct return code) is still a real crash.
        if run.returncode < 0 or run.returncode >= 128:
            raise RuntimeError(
                f"bc-iclib execution failed with exit {run.returncode}: "
                f"{run.stdout}{run.stderr}".strip()
            )
        with open("output.csv", "w", encoding="utf-8") as output:
            output.write(run.stdout)
        if not run.stdout.strip():
            raise RuntimeError("bc-iclib execution produced no output.csv")
        return _normalized(
            calculate_mape_from_csv("output.csv", "ground_truth.csv")
        )
    finally:
        os.chdir(cwd)

# ---------------------------------------------------------------------------
#  Shared evaluator dispatch
# ---------------------------------------------------------------------------

#: Benchmark name -> error evaluator. Previously this mapping existed only as
#: an if/elif chain inside ``lib/bo.py``, so nothing else could measure a
#: candidate's error without duplicating it.
ERROR_EVALUATORS = {
    "susan": lambda p: susanError(p),
    "sobel-iclib": lambda p: sobelError(p),
    "accept-sobel": lambda p: acceptSobelError(p),
    "lqi-iclib": lambda p: lqiError(p),
    "link-estimator": lambda p: lqiError(p),
    "stringsearch-iclib": lambda p: stringSearchError(p),
    "fft-iclib": lambda p: fftError(p),
    "ar-iclib": lambda p: arError(p),
    "accept-activityrec": lambda p: acceptActivityrecError(p),
    "bc-iclib": lambda p: bitcountError(p),
    "radix-bm": lambda p: radixbmError(p),
    "segment-bm": lambda p: segmentbmError(p),
}


def evaluate_error(app_name, path):
    """Measure *app_name*'s error for the candidate built in *path*."""
    evaluator = ERROR_EVALUATORS.get(app_name)
    if evaluator is None:
        raise RuntimeError(f"no error evaluator registered for {app_name!r}")
    return evaluator(path)


#: Files that carry the reference output for each benchmark family.
GROUND_TRUTH_ARTIFACTS = {
    "susan": ["out.pgm"],
    "sobel-iclib": ["out.pgm"],
    "sobel-esp": ["out.pgm"],
    "accept-sobel": ["out.pgm"],
    "stringsearch-iclib": ["ground_truth.csv"],
    "ar-iclib": ["ground_truth.csv"],
    "accept-activityrec": ["ground_truth.csv"],
    "fft-iclib": ["ground_truth.csv"],
    "bc-iclib": ["ground_truth.csv"],
    "radix-bm": ["ground_truth.csv"],
    "segment-bm": ["ground_truth.csv"],
    "lqi-iclib": ["ground_truth.txt"],
    "link-estimator": ["ground_truth.txt"],
}


def ground_truth_files(app_name):
    """Return the reference artifacts *app_name* is scored against."""
    return list(GROUND_TRUTH_ARTIFACTS.get(app_name, []))

