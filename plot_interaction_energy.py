"""Plot the sum of reconstructed particle energies for each DLP interaction."""

import argparse
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import uproot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nominal", help="Nominal ROOT file")
    parser.add_argument("up", help="Up variation ROOT file")
    parser.add_argument("down", help="Down variation ROOT file")
    parser.add_argument("-n", "--name", default="caf", help="RNTuple name")
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("-o", "--output", default="interaction_energy.png")
    args = parser.parse_args()
    if args.bins <= 0:
        parser.error("--bins must be positive")

    energies = []
    labels = ["Nominal", "Up", "Down"]
    for label, filename in zip(labels, [args.nominal, args.up, args.down]):
        with uproot.open(filename) as root_file:
            records = root_file[args.name].arrays()
            particles = records.rec.common.ixn.dlp.part.dlp
            # Sum the particle axis before flattening the event/interaction axes.
            interaction_energy = ak.flatten(ak.sum(particles.E, axis=-1), axis=None)
            energies.append(ak.to_numpy(interaction_energy))
        print(f"{label}: {len(interaction_energy)} interactions from {filename}")

    output = Path(args.output)
    limited_output = output.with_name(f"{output.stem}_0_30{output.suffix}")
    limiter_output = output.with_name(f"{output.stem}_0_3{output.suffix}")
    for plot_output, energy_range, title in [
        (output, None, "Reconstructed interaction energy"),
        (limited_output, (0, 30), "Reconstructed interaction energy (0-30)"),
        (limiter_output, (0, 3), "Reconstructed interaction energy (0-3)"),
    ]:
        fig, ax = plt.subplots()
        ax.hist(energies, bins=args.bins, range=energy_range, histtype="step", label=labels)
        ax.set_xlabel("Total reconstructed interaction energy (CAF energy units)")
        ax.set_ylabel("Interactions")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_output)
        plt.close(fig)
        print(f"Saved histogram to {plot_output}")


if __name__ == "__main__":
    main()
