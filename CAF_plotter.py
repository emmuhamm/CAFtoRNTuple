"""Read selected CAF branches, derive interaction quantities, and plot variations."""

import argparse
from pathlib import Path
import numpy as np

import awkward as ak
import matplotlib.pyplot as plt
import uproot






class CAFFile:
    """Lazy, read-only access to a CAF file with in-memory derived data."""

    _PARTICLE_PATH = "rec.common.ixn.dlp.part.dlp"
    _ND_LAR_DLP_PATH = "rec.nd.lar.dlp"

    def __init__(self, file_path, tree_name="caf"):
        self.file_path = Path(file_path)
        self.tree_name = tree_name
        self._particle_energy = f"{self._PARTICLE_PATH}.E"
        self._particle_pdg = f"{self._PARTICLE_PATH}.pdg"
        self._nd_track_energy = f"{self._ND_LAR_DLP_PATH}.tracks.E"
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
            "interaction_energy",
            self._calculate_interaction_energy,
            aliases=("rec.common.ixn.dlp.E",),
        )
        self.register_derived(
            "interaction_energy_fractional_difference",
            self._calculate_interaction_energy_fractional_difference,
            aliases=("rec.common.ixn.dlp.energy_fractional_difference",),
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
        self.register_derived(
            "e_shower",
            self._calculate_e_shower,
            aliases=("rec.common.ixn.dlp.part.e_shower",),
        )
        self.register_derived(
            "e_shower_interaction",
            self._calculate_e_shower_interaction,
            aliases=("e_shower_full", "rec.common.ixn.dlp.e_shower_full"),
        )
        self.register_derived(
            "total_energy",
            self._calculate_total_energy,
            aliases=(f"{self._ND_LAR_DLP_PATH}.total_energy",),
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

    def interaction_energy(self):
        """Return summed particle energy for each reconstructed interaction."""
        return self.get_data("interaction_energy")

    def interaction_energy_fractional_difference(self):
        """Return (reconstructed - true) energy divided by true energy."""
        return self.get_data("interaction_energy_fractional_difference")

    def part_length(self):
        """Return per-particle reconstructed lengths, with non-tracks set to 0."""
        return self.get_data("part_length")

    def part_max_length(self):
        """Return the longest reconstructed track in each DLP interaction."""
        return self.get_data("part_max_length")

    def e_shower(self):
        """Return energies of electron/positron and photon candidates."""
        return self.get_data("e_shower")

    def e_shower_interaction(self):
        """Return the summed shower-candidate energy for each interaction."""
        return self.get_data("e_shower_interaction")

    def total_energy(self):
        """Return the summed track energy for each ND-LAr DLP interaction."""
        return self.get_data("total_energy")

    def _calculate_total_energy(self):
        return ak.sum(self.get_data(self._nd_track_energy), axis=-1)

    def _calculate_e_shower(self):
        fields = self._read_fields([self._particle_energy, self._particle_pdg])
        pdg = abs(fields[self._particle_pdg])
        return fields[self._particle_energy][(pdg == 11) | (pdg == 22)]

    def _calculate_e_shower_interaction(self):
        return ak.sum(self.get_data("e_shower"), axis=-1)

    def _calculate_interaction_energy(self):
        energies = self._read_fields([self._particle_energy])[self._particle_energy]
        return ak.sum(energies, axis=-1)

    def _calculate_interaction_energy_fractional_difference(self):
        reconstructed_energy = self.get_data("rec.common.ixn.dlp.E")
        truth_indices = self.get_data("rec.common.ixn.dlp.truth")
        truth_overlap = self.get_data("rec.common.ixn.dlp.truthOverlap")
        true_energy = self.get_data("rec.mc.nu.E")

        best_match = ak.argmax(
            truth_overlap, axis=-1, keepdims=True, mask_identity=True
        )
        best_truth_index = ak.firsts(truth_indices[best_match], axis=-1)
        best_truth_index = ak.mask(
            best_truth_index, best_truth_index < ak.num(true_energy, axis=1)
        )
        matched_true = ak.Array(
            [
                spill_energy[spill_truth_indices]
                for spill_energy, spill_truth_indices in zip(
                    true_energy, best_truth_index
                )
            ]
        )
        valid = (
            np.isfinite(reconstructed_energy)
            & np.isfinite(matched_true)
            & (matched_true != 0)
        )
        valid = ak.fill_none(valid, False)
        fractional_difference = (reconstructed_energy - matched_true) / matched_true
        return fractional_difference[valid]

    def _calculate_part_length(self):
        paths = [self._particle_pdg, *self._start_fields, *self._end_fields]
        fields = self._read_fields(paths)

        start = [fields[path] for path in self._start_fields]
        end = [fields[path] for path in self._end_fields]
        distance_squared = sum(
            (end_axis - start_axis) ** 2
            for start_axis, end_axis in zip(start, end)
        )
        lengths = distance_squared**0.5
        pdg = abs(fields[self._particle_pdg])

        # Exclude electron/positron and photon candidates using reconstructed PDG.
        valid = (
            np.isfinite(lengths)
            & (lengths > 0)
            & (pdg != 11)
            & (pdg != 22)
        )
        cleaned = ak.where(valid, lengths, 0)

        # SPINE leaves origRecoObjType as -1; recoobj.type may identify tracks
        # if its enum is verified against the CAF production version.
        return cleaned

        
        # return ak.where(np.isfinite(lengths) & (lengths > 0), lengths, 0)

    def _calculate_part_max_length(self):
        lengths = self.get_data("part_length")
        maximum = ak.max(lengths, axis=-1, mask_identity=True)
        maximum = ak.fill_none(maximum, 0)
        ret = ak.flatten(maximum, axis=None)
        return ret

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
    "true_neutrino_energy": {
        "file_name": "true_neutrino_energy",
        "x_axis": [0, -1],
        # "percentile": 95,
        "y_axis": [0, -1],
        "x_log": False,
        "y_log": False,
        "x_title": "True neutrino energy (GeV)",
        "y_title": "True interactions",
        "x_data": "rec.mc.nu.E",
        "bins": 50,
    },
    "track_energy": {
        "file_name": "track_energy",
        "x_axis": [0, -1],
        "percentile": 95,
        "y_axis": [0, -1],
        "x_log": False,
        "y_log": False,
        "x_title": "ND-LAr DLP track energy (GeV)",
        "y_title": "Tracks",
        "x_data": "rec.nd.lar.dlp.tracks.E",
        "bins": 50,
    },
    "total_energy": {
        "file_name": "total_energy",
        "x_axis": [0, -1],
        "percentile": 95,
        "y_axis": [0, -1],
        "x_log": False,
        "y_log": False,
        "x_title": "Total ND-LAr DLP track energy per interaction (GeV)",
        "y_title": "Interactions",
        "x_data": "rec.nd.lar.dlp.total_energy",
        "bins": 50,
    },
    "e_shower": {
        "file_name": "e_shower",
        "x_axis": [0.0001, 0.02],
        # "percentile": 95,
        "y_axis": [0, -1],
        "x_log": False,
        "y_log": False,
        "x_title": "Reconstructed shower-candidate energy (GeV)",
        "y_title": "Shower candidates",
        "x_data": "e_shower",
        "bins": 50,
    },
    "e_shower_interaction": {
        "file_name": "e_shower_interaction",
        "x_axis": [0.0001, 0.02],
        # "percentile": 95,
        "y_axis": [0, -1],
        "x_log": False,
        "y_log": False,
        "x_title": "Total reconstructed shower energy per interaction (GeV)",
        "y_title": "Interactions",
        "x_data": "e_shower_interaction",
        "bins": 50,
    },
    "interaction_energy": {
        "file_name": "interaction_energy",
        "x_axis": [0, -1],
        "y_axis": [0, -1],
        "x_log": False,
        "y_log": False,
        "x_title": "Total reconstructed interaction energy (GeV)",
        "y_title": "Interactions",
        "x_data": "rec.common.ixn.dlp.E",
        "bins": 50,
    },
    "interaction_energy_fractional_difference": {
        "file_name": "interaction_energy_fractional_difference",
        "x_axis": [-2,  1],
        "y_axis": [0, -1],
        "x_log": False,
        "y_log": False,
        "x_title": "(Reconstructed interaction energy - true interaction energy) / true interaction energy",
        "y_title": "(Doesnt work!) Interactions",
        "x_data": "interaction_energy_fractional_difference",
        "bins": 50,
    },
    
    "interaction_energy_30": {
        "file_name": "interaction_energy_3_log",
        "x_axis": [0, 3],
        "y_axis": [0, -1],
        "x_log": False,
        "y_log": True,
        "x_title": "Total reconstructed interaction energy (GeV)",
        "y_title": "Interactions",
        "x_data": "rec.common.ixn.dlp.E",
        "bins": 50,
    },
    
    "part_max_length": {
        "file_name": "part_max_length",
        "x_axis": [0, 5],
        "y_axis": [0.01, -1],
        # "percentile": 80,
        "x_log": False,
        "y_log": False,
        "x_title": "Longest reconstructed track length per interaction (cm)",
        "y_title": "Interactions",
        "x_data": "part_max_length",
        "bins": 50,
    },

    "part_max_length_log": {
        "file_name": "part_max_length_log",
        "x_axis": [0, 5],
        "y_axis": [0.01, -1],
        # "percentile": 80,
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

    @staticmethod
    def _percentile_range(config, nominal):
        limits = config.get("x_axis")
        if limits is None or len(limits) != 2 or limits[1] != -1:
            raise ValueError("percentile requires an x_axis upper limit of -1")
        percentile = config["percentile"]
        if (
            not isinstance(percentile, (int, float))
            or isinstance(percentile, bool)
            or not 0 < percentile <= 100
        ):
            raise ValueError("percentile must be a number greater than 0 and at most 100")

        finite = nominal[np.isfinite(nominal)]
        if len(finite) == 0:
            raise ValueError("Cannot calculate a percentile without finite nominal data")
        upper = np.percentile(finite, percentile)
        lower = limits[0]
        if lower is None or lower == -1:
            lower = finite.min()
        if config.get("x_log", False) and lower <= 0:
            positive = finite[finite > 0]
            if len(positive) == 0:
                raise ValueError("A logarithmic x axis requires positive nominal data")
            lower = positive.min()
        if not np.isfinite(lower) or upper <= lower:
            raise ValueError(
                "The nominal percentile must exceed the lower x limit; "
                "choose a higher percentile or a manual range"
            )
        return lower, upper

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

        if "percentile" in config:
            histogram_range = self._percentile_range(config, values[0])
            config = {**config, "x_axis": histogram_range}
        else:
            histogram_range = self._linear_histogram_range(config)

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
            range=histogram_range,
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
        # ratio_ax.legend()

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

if __name__ == "__main__":
    main()
else:
    print("heh")