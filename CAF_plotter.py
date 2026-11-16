"""Read selected CAF branches, derive interaction quantities, and plot variations."""

import argparse
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import uproot


_PARTICLE_PATH = "rec.common.ixn.dlp.part.dlp"
_PARTICLE_ENERGY = f"{_PARTICLE_PATH}.E"
_TRACK_TYPE = f"{_PARTICLE_PATH}.origRecoObjType"
_START_FIELDS = [f"{_PARTICLE_PATH}.start.{axis}" for axis in "xyz"]
_END_FIELDS = [f"{_PARTICLE_PATH}.end.{axis}" for axis in "xyz"]

_DERIVED_ALIASES = {
    "part_E_sum": "part_E_sum",
    "rec.common.ixn.dlp.part_E_sum": "part_E_sum",
    "part_length": "part_length",
    f"{_PARTICLE_PATH}.part_length": "part_length",
    "part_max_length": "part_max_length",
    "rec.common.ixn.dlp.part_max_length": "part_max_length",
}


class CAFFile:
    """Lazy, read-only access to a CAF file with in-memory derived data."""

    def __init__(self, file_path, tree_name="caf"):
        self.file_path = Path(file_path)
        self.tree_name = tree_name
        self._branch_cache = {}
        self._derived_cache = {}

    def _read_fields(self, paths):
        missing = [path for path in paths if path not in self._branch_cache]
        if missing:
            with uproot.open(self.file_path) as root_file:
                tree = root_file[self.tree_name]
                if hasattr(tree, "header"):
                    arrays = tree.arrays(filter_name=missing)
                else:
                    arrays = tree.arrays(filter_name=missing, library="ak")

            for path in missing:
                value = arrays
                for field in path.split("."):
                    value = value[field]
                self._branch_cache[path] = value
        return {path: self._branch_cache[path] for path in paths}

    def _particle_lengths(self):
        if "part_length" not in self._derived_cache:
            paths = [_TRACK_TYPE, *_START_FIELDS, *_END_FIELDS]
            fields = self._read_fields(paths)

            start = [fields[path] for path in _START_FIELDS]
            end = [fields[path] for path in _END_FIELDS]
            distance_squared = sum(
                (end_axis - start_axis) ** 2
                for start_axis, end_axis in zip(start, end)
            )
            lengths = distance_squared**0.5
            self._derived_cache["part_length"] = ak.where(
                fields[_TRACK_TYPE] == 1, lengths, 0
            )
        return self._derived_cache["part_length"]

    def part_E_sum(self):
        """Return the summed particle energy for each DLP interaction."""
        if "part_E_sum" not in self._derived_cache:
            energies = self._read_fields([_PARTICLE_ENERGY])[_PARTICLE_ENERGY]
            interaction_energy = ak.sum(energies, axis=-1)
            self._derived_cache["part_E_sum"] = ak.flatten(
                interaction_energy, axis=None
            )
        return self._derived_cache["part_E_sum"]

    def part_length(self):
        """Return per-particle reconstructed lengths, with non-tracks set to 0."""
        return self._particle_lengths()

    def part_max_length(self):
        """Return the longest reconstructed track in each DLP interaction."""
        if "part_max_length" not in self._derived_cache:
            lengths = self._particle_lengths()
            maximum = ak.max(lengths, axis=-1, mask_identity=True)
            maximum = ak.fill_none(maximum, 0)
            self._derived_cache["part_max_length"] = ak.flatten(
                maximum, axis=None
            )
        return self._derived_cache["part_max_length"]

    def get_data(self, path):
        """Get a raw dotted CAF path or one of the in-memory derived values."""
        derived_name = _DERIVED_ALIASES.get(path)
        if derived_name == "part_E_sum":
            return self.part_E_sum()
        if derived_name == "part_length":
            return self.part_length()
        if derived_name == "part_max_length":
            return self.part_max_length()
        return self._read_fields([path])[path]


PLOT_CONFIGS = {
    "interaction_energy": {
        "file_name": "interaction_energy",
        "x_axis": [0, 30],
        "y_axis": [0, -1],
        "x_log": False,
        "y_log": False,
        "x_title": "Total reconstructed interaction energy (GeV)",
        "y_title": "Interactions",
        "x_data": "part_E_sum",
        "bins": 50,
    },
    "part_max_length": {
        "file_name": "part_max_length",
        "x_axis": [0, -1],
        "y_axis": [0, -1],
        "x_log": False,
        "y_log": True,
        "x_title": "Longest reconstructed track length per interaction (cm)",
        "y_title": "Interactions",
        "x_data": "part_max_length",
        "bins": 50,
    },
}


class Plotter:
    """Compare the same CAF-derived quantity across nominal/up/down files."""

    def __init__(
        self,
        caf_files,
        output_dir="plots",
        heading="",
        colors=("black", "red", "blue"),
        bins=50,
    ):
        if len(caf_files) != 3:
            raise ValueError("Exactly three CAF files are required: nominal, up, down")
        if len(colors) != 3:
            raise ValueError("Exactly three colors are required: nominal, up, down")
        if bins <= 0:
            raise ValueError("bins must be positive")

        self.caf_files = dict(zip(("Nominal", "Up", "Down"), caf_files))
        self.output_dir = Path(output_dir)
        self.heading = heading
        self.colors = colors
        self.bins = bins

    @staticmethod
    def _histogram_values(data):
        if ak.fields(data):
            raise TypeError("Plot data must be a numeric CAF field, not a record")
        return ak.to_numpy(ak.flatten(data, axis=None))

    @staticmethod
    def _set_axis_limits(axes, axis_name, limits, logarithmic):
        if limits is None:
            return
        if len(limits) != 2:
            raise ValueError("Axis limits must contain a lower and upper bound")

        lower, upper = limits
        lower = None if lower is None or lower == -1 else lower
        upper = None if upper is None or upper == -1 else upper
        if logarithmic and lower is not None and lower <= 0:
            lower = None

        if lower is not None or upper is not None:
            if axis_name == "x":
                axes.set_xlim(left=lower, right=upper)
            else:
                axes.set_ylim(bottom=lower, top=upper)

    def plot(self, config):
        x_data = config["x_data"]
        values = []
        labels = list(self.caf_files)
        for caf_file in self.caf_files.values():
            values.append(
                self._histogram_values(caf_file.get_data(x_data))
            )

        bins = config.get("bins", self.bins)
        if bins <= 0:
            raise ValueError("Plot config bins must be positive")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        file_name = config["file_name"]
        output = self.output_dir / f"{self.heading}{file_name}.png"

        fig, ax = plt.subplots()
        ax.hist(
            values,
            bins=bins,
            range=self._linear_histogram_range(config),
            histtype="step",
            label=labels,
            color=self.colors,
        )
        if config.get("x_log", False):
            ax.set_xscale("log")
        if config.get("y_log", False):
            ax.set_yscale("log")
        ax.set_xlabel(config["x_title"])
        ax.set_ylabel(config["y_title"])
        ax.set_title(config.get("title", file_name.replace("_", " ").title()))
        ax.legend()

        self._set_axis_limits(
            ax, "x", config.get("x_axis"), config.get("x_log", False)
        )
        self._set_axis_limits(
            ax, "y", config.get("y_axis"), config.get("y_log", False)
        )
        fig.tight_layout()
        fig.savefig(output)
        plt.close(fig)
        print(f"Saved histogram to {output}")
        return output

    @staticmethod
    def _linear_histogram_range(config):
        limits = config.get("x_axis")
        if limits is None or config.get("x_log", False):
            return None
        if len(limits) != 2:
            raise ValueError("x_axis must contain a lower and upper bound")
        lower, upper = limits
        if lower is None or upper is None or lower == -1 or upper == -1:
            return None
        return lower, upper

    def plot_all(self, configs):
        return [self.plot(config) for config in configs.values()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nominal", help="Nominal CAF ROOT file")
    parser.add_argument("up", help="Up-variation CAF ROOT file")
    parser.add_argument("down", help="Down-variation CAF ROOT file")
    parser.add_argument("-n", "--name", default="caf", help="RNTuple/tree name")
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("-o", "--output-dir", default="plots")
    parser.add_argument("--heading", default="", help="Prefix for plot file names")
    parser.add_argument(
        "--colors",
        nargs=3,
        default=("black", "red", "blue"),
        metavar=("NOMINAL", "UP", "DOWN"),
        help="Histogram colors for the three input files",
    )
    parser.add_argument(
        "--plot",
        choices=tuple(PLOT_CONFIGS),
        action="append",
        help="Plot config to run; repeat to select multiple (default: all)",
    )
    args = parser.parse_args()
    if args.bins <= 0:
        parser.error("--bins must be positive")

    caf_files = [
        CAFFile(args.nominal, args.name),
        CAFFile(args.up, args.name),
        CAFFile(args.down, args.name),
    ]
    plotter = Plotter(
        caf_files,
        output_dir=args.output_dir,
        heading=args.heading,
        colors=args.colors,
        bins=args.bins,
    )
    selected = args.plot or list(PLOT_CONFIGS)
    plotter.plot_all({name: PLOT_CONFIGS[name] for name in selected})


if __name__ == "__main__":
    main()
