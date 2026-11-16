"""Read selected CAF branches, derive interaction quantities, and plot variations."""

import argparse
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import uproot


class CAFFile:
    """Lazy, read-only access to a CAF file with in-memory derived data."""

    _PARTICLE_PATH = "rec.common.ixn.dlp.part.dlp"

    def __init__(self, file_path, tree_name="caf"):
        self.file_path = Path(file_path)
        self.tree_name = tree_name
        self._particle_energy = f"{self._PARTICLE_PATH}.E"
        self._track_type = f"{self._PARTICLE_PATH}.origRecoObjType"
        self._start_fields = [
            f"{self._PARTICLE_PATH}.start.{axis}" for axis in "xyz"
        ]
        self._end_fields = [
            f"{self._PARTICLE_PATH}.end.{axis}" for axis in "xyz"
        ]
        self._branch_cache = {}
        self._derived_cache = {}
        self._derived_calculations = {}
        self._derived_aliases = {}
        self.register_derived(
            "part_E_sum",
            self._calculate_part_E_sum,
            aliases=("rec.common.ixn.dlp.part_E_sum",),
        )
        self.register_derived(
            "part_length",
            self._calculate_part_length,
            aliases=(f"{self._PARTICLE_PATH}.part_length",),
        )
        self.register_derived(
            "part_max_length",
            self._calculate_part_max_length,
            aliases=("rec.common.ixn.dlp.part_max_length",),
        )

    def register_derived(self, name, calculation, aliases=()):
        """Register a lazy, cached calculation under a name and optional paths."""
        if not isinstance(name, str) or not name:
            raise ValueError("A derived-data name must be a non-empty string")
        if not callable(calculation):
            raise TypeError("A derived-data calculation must be callable")
        if isinstance(aliases, str):
            aliases = (aliases,)

        all_aliases = (name, *aliases)
        if any(not isinstance(alias, str) or not alias for alias in all_aliases):
            raise ValueError("Derived-data aliases must be non-empty strings")
        unique_aliases = tuple(dict.fromkeys(all_aliases))
        conflicts = [alias for alias in unique_aliases if alias in self._derived_aliases]
        if name in self._derived_calculations or conflicts:
            raise ValueError(f"Derived-data name or alias is already registered: {name}")

        self._derived_calculations[name] = calculation
        for alias in unique_aliases:
            self._derived_aliases[alias] = name

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

    def part_E_sum(self):
        """Return the summed particle energy for each DLP interaction."""
        return self.get_data("part_E_sum")

    def part_length(self):
        """Return per-particle reconstructed lengths, with non-tracks set to 0."""
        return self.get_data("part_length")

    def part_max_length(self):
        """Return the longest reconstructed track in each DLP interaction."""
        return self.get_data("part_max_length")

    def _calculate_part_E_sum(self):
        energies = self._read_fields([self._particle_energy])[self._particle_energy]
        interaction_energy = ak.sum(energies, axis=-1)
        return ak.flatten(interaction_energy, axis=None)

    def _calculate_part_length(self):
        paths = [self._track_type, *self._start_fields, *self._end_fields]
        fields = self._read_fields(paths)

        start = [fields[path] for path in self._start_fields]
        end = [fields[path] for path in self._end_fields]
        distance_squared = sum(
            (end_axis - start_axis) ** 2
            for start_axis, end_axis in zip(start, end)
        )
        lengths = distance_squared**0.5
        return ak.where(fields[self._track_type] == 1, lengths, 0)

    def _calculate_part_max_length(self):
        lengths = self.get_data("part_length")
        maximum = ak.max(lengths, axis=-1, mask_identity=True)
        maximum = ak.fill_none(maximum, 0)
        return ak.flatten(maximum, axis=None)

    def get_data(self, path):
        """Get a raw dotted CAF path or one of the in-memory derived values."""
        derived_name = self._derived_aliases.get(path)
        if derived_name is not None:
            if derived_name not in self._derived_cache:
                self._derived_cache[derived_name] = self._derived_calculations[
                    derived_name
                ]()
            return self._derived_cache[derived_name]
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
    def _ratios_to_nominal(histogram_counts):
        nominal_counts = histogram_counts[0]
        return [
            [
                count / nominal if nominal else float("nan")
                for count, nominal in zip(variation_counts, nominal_counts)
            ]
            for variation_counts in histogram_counts[1:]
        ]

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

        fig, (ax, ratio_ax) = plt.subplots(
            2,
            1,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        histogram_counts, bin_edges, _ = ax.hist(
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
        ax.set_ylabel(config["y_title"])
        ax.set_title(config.get("title", file_name.replace("_", " ").title()))
        ax.legend()

        bin_centers = [
            (left + right) / 2 for left, right in zip(bin_edges[:-1], bin_edges[1:])
        ]
        ratios = self._ratios_to_nominal(histogram_counts)
        for label, color, ratio in zip(labels[1:], self.colors[1:], ratios):
            ratio_ax.step(
                bin_centers,
                ratio,
                where="mid",
                label=f"{label} / Nominal",
                color=color,
            )
        ratio_ax.axhline(1, color="black", linestyle="--", linewidth=1)
        ratio_ax.set_xlabel(config["x_title"])
        ratio_ax.set_ylabel(config.get("ratio_title", "Variation / Nominal"))
        ratio_ax.legend()

        self._set_axis_limits(
            ax, "x", config.get("x_axis"), config.get("x_log", False)
        )
        self._set_axis_limits(
            ax, "y", config.get("y_axis"), config.get("y_log", False)
        )
        self._set_axis_limits(
            ratio_ax, "y", config.get("ratio_axis"), False
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
