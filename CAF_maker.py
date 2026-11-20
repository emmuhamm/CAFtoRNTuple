"""Make normalised CAF histograms and their variation weights."""

import argparse

import numpy as np
import uproot

from CAF_plotter import CAFFile, Plotter


SAMPLES = ("nominal", "up", "down")


def _get_bin_edges(bins, value_range, bin_edges=None):
    if bin_edges is None:
        lower, upper = value_range
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("The histogram range must be finite and increasing")
        if bins <= 0:
            raise ValueError("bins must be positive")
        bin_edges = np.linspace(lower, upper, bins + 1)
    else:
        bin_edges = np.asarray(bin_edges, dtype=float)
        if (
            bin_edges.ndim != 1
            or len(bin_edges) < 2
            or not np.all(np.isfinite(bin_edges))
            or np.any(np.diff(bin_edges) <= 0)
        ):
            raise ValueError("bin_edges must be finite and strictly increasing")
    return bin_edges


def make_histograms(caf_files, x_data, bins, value_range, bin_edges=None):
    """Return area-normalised histograms, bin edges, and variation weights."""
    bin_edges = _get_bin_edges(bins, value_range, bin_edges)

    histograms = {}
    for sample, caf_file in zip(SAMPLES, caf_files):
        values = Plotter._histogram_values(caf_file.get_data(x_data))
        values = values[np.isfinite(values)]
        counts, _ = np.histogram(values, bins=bin_edges)
        area = np.sum(counts * np.diff(bin_edges))
        if area == 0:
            raise ValueError(
                f"Cannot normalise {sample}: no finite entries in histogram range"
            )
        histograms[sample] = counts / area

    nominal = histograms["nominal"]
    weights = {
        sample: np.divide(
            histogram,
            nominal,
            out=np.zeros_like(histogram),
            where=nominal != 0,
        )
        for sample, histogram in histograms.items()
    }
    return histograms, bin_edges, weights


def save_plots(plotter, histograms, bin_edges, weights, x_data):
    histogram_output = plotter.plot_histogram_arrays(
        [histograms[sample] for sample in SAMPLES],
        bin_edges,
        "histograms.png",
        x_data,
        "Normalised CAF histograms",
    )
    weights_output = plotter.plot_weight_arrays(
        [weights[sample] for sample in SAMPLES],
        bin_edges,
        "weights.png",
        x_data,
        "Variation weights",
    )
    return histogram_output, weights_output


def save_root_file(histograms, bin_edges, weights, output_path):
    with uproot.recreate(output_path) as root_file:
        for sample in SAMPLES:
            root_file[f"histograms/A_{sample}"] = (histograms[sample], bin_edges)
            root_file[f"weights/A_{sample}"] = (weights[sample], bin_edges)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nominal", help="Nominal CAF ROOT file")
    parser.add_argument("up", help="Up-variation CAF ROOT file")
    parser.add_argument("down", help="Down-variation CAF ROOT file")
    parser.add_argument("-n", "--name", default="caf", help="RNTuple/tree name")
    parser.add_argument("--x-data", default="interaction_energy")
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument(
        "--bin-edges",
        type=float,
        nargs="+",
        metavar="EDGE",
        help="Explicit histogram bin edges; takes precedence over --bins and --range",
    )
    parser.add_argument(
        "--range",
        dest="value_range",
        type=float,
        nargs=2,
        default=(0.0, 3.0),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument("--output-root", default="caf_histograms.root")
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()
    if args.bin_edges is None and args.bins <= 0:
        parser.error("--bins must be positive")

    try:
        bin_edges = _get_bin_edges(
            args.bins, args.value_range, args.bin_edges
        )
    except ValueError as error:
        parser.error(str(error))

    caf_files = [
        CAFFile(args.nominal, args.name),
        CAFFile(args.up, args.name),
        CAFFile(args.down, args.name),
    ]
    plotter = Plotter(
        caf_files,
        output_dir=args.output_dir,
        bins=len(bin_edges) - 1 if args.bin_edges is not None else args.bins,
    )
    histograms, bin_edges, weights = make_histograms(
        caf_files, args.x_data, args.bins, args.value_range, bin_edges
    )
    save_root_file(histograms, bin_edges, weights, args.output_root)
    plot_paths = save_plots(
        plotter, histograms, bin_edges, weights, args.x_data
    )
    print(f"Saved ROOT histograms and weights to {args.output_root}")
    for plot_path in plot_paths:
        print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()