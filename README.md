# CAFtoRNTuple

This is a simple utility to convert a TTree CAF into an RNTuple CAF. The structure should otherwise be identical.

To build:
```
mkdir build && cd build
cmake .. && make install
```

And then to perform the conversion (from within the build directory):
```
./converter rntuple.root ttree_caf_1.root ttree_caf_2.root ...
```

This recipe works in an el9 or MacOS environment without duneanaobj installed already. It has not been tested in an environment where duneanaobj is already installed. Please report issues if they arise.

Currently, the `nd.sand` branches are turned off. If left on, segfault arise, at least when running on the MicroProdN4p1 CAFs. The origin of the segfaults has not yet been tracked down.

## Speed comparisons

There are additional scripts that are designed to test how fast it is to read in and perform a simple operation on the TTree and RNTuple CAFs. The simple operation is filling a histogram of muon energies.

After converting 60 files from MicroProdN4p1 to RNTuple, the scripts report the following times:
1. TTree (./build/test_ttree): 1.3s
2. RNTuple: (./build/test_rntuple): 1.14s
3. RNTuple but only reading the common reco branch (./build/test_rntuple_better): 0.48s
4. Python w/ uproot (test_rntuple_uproot.py): 1.35s
5. Python w/ uproot but only reading dlp particles (test_rntuple_uproot.py): 0.08s

These are not a comprehensive set of tests. I suspect there are better ways to do things. However, I do think that the simplest way to make things quick currently is with uproot and rntuple (explore [test_rntuple_uproot.py](test_rntuple_uproot.py)).

## Plotting CAF variations

`CAF_plotter.py` reads selected branches from three CAF files in nominal, up,
down order. CAF files are read-only; derived quantities are computed on demand
and cached in memory. Each plot has a top histogram comparison and a bottom
variation-to-nominal ratio panel, using shared histogram bins. Bins with zero
nominal entries are omitted from the ratio. By default, it plots the summed
DLP particle energy, the longest reconstructed track length per interaction,
individual and summed shower-candidate energies, and individual and summed
ND-LAr DLP track energies, plus true neutrino energies:

Each Nominal/Up/Down histogram is independently fitted with a Crystal Ball
distribution. Dashed curves show the fits, and dotted vertical lines show the
fitted means. Fits are skipped when there are fewer than five finite entries in
the plotted range or the fit fails; a mean marker is omitted if the fitted
Crystal Ball distribution has no finite mean.

For `e_shower_interaction`, `x_axis` controls the displayed histogram range,
while `fit_range` selects the values used in the fit. `fit_bounds` accepts
`beta`, `m`, `mean`, and `scale` pairs; `mean` bounds constrain the actual
Crystal Ball mean, not SciPy's location parameter. Keeping the lower `m` bound
above 2 ensures a finite fitted mean.

Set `"normalize": True` in any `PLOT_CONFIGS` entry to give each sample's
histogram unit area over the histogram bin range. The y axis becomes
probability density, fit curves receive the same scaling, and the ratio panel
compares normalised shapes. Empty in-range samples raise an error. The default
is `False`, preserving raw interaction counts.

```bash
python CAF_plotter.py nominal.root up.root down.root --output-dir plots
```

Select one or more plots with repeatable `--plot` options, and optionally set
an output filename prefix with `--heading`:

```bash
python CAF_plotter.py nominal.root up.root down.root \
  --plot interaction_energy --heading shifted_ --output-dir plots
```

The plot definitions are in `PLOT_CONFIGS`. `CAFFile.get_data()` accepts either
a raw dotted CAF branch path or a supported derived quantity such as
`interaction_energy` or `part_max_length`.

The `interaction_energy_fractional_difference` plot calculates
`(reconstructed interaction energy - true interaction energy) / true interaction
energy`. For each reconstructed interaction, it uses the truth index with the
largest corresponding `rec.common.ixn.dlp.truthOverlap` value to select an entry
from `rec.mc.nu`. Interactions without a truth match, with an out-of-range truth
index, or with zero or non-finite energy are excluded. Select it with
`--plot interaction_energy_fractional_difference`.

The `true_neutrino_energy` plot reads `rec.mc.nu.E`, with one entry per true
interaction, using the nominal 95th-percentile cutoff and the ratio panel.
Select it with `--plot true_neutrino_energy`.

The `e_shower` quantity selects particle energies with `abs(pdg)` equal to 11
(electron/positron) or 22 (photon), excluding all other particles while retaining
the event/interaction structure. `e_shower_interaction` sums those energies per
interaction, including zero for interactions without matching particles. These
are PDG-based shower candidates, not a reconstruction-shape classification.
Their dotted aliases are `rec.common.ixn.dlp.part.e_shower` and
`rec.common.ixn.dlp.e_shower_full`, respectively. Both plot configs include the
existing variation-to-nominal ratio panel.

The `track_energy` plot reads `rec.nd.lar.dlp.tracks.E`. The `total_energy`
quantity (alias `rec.nd.lar.dlp.total_energy`) sums these track energies per
ND-LAr DLP interaction, retaining zero for interactions with no tracks. Both
energy plots use the nominal 95th percentile and the existing ratio panel.
The sums use the stored energies without additional particle selections or
value cuts. Charge plots and totals are not implemented because `tracks.charge`
is uniformly zero in the checked nominal sample.

Set `"percentile": 95` with `"x_axis": [0, -1]` to restrict the shared histogram
bins and x axis to the lowest approximately 95% of finite nominal entries.
Both shower plots enable this by default. A percentile is only accepted when
the upper x limit is `-1`; manual upper limits cannot be combined with it.
All three samples use the nominal cutoff, without renormalisation. Empty
nominal data or a cutoff at/below the lower limit raises an explicit error.

Additional in-memory quantities can be added with
`CAFFile.register_derived(name, calculation, aliases=())`. The calculation is a
zero-argument callable, run on the first request and cached for that `CAFFile`
instance; aliases let it also be requested using a dotted CAF-style path.
