# First look at detector systematics

This is some hacky shit but its a good first look for the systematics. Bare with me..

Note that all of this runs on the GPVMs so far

We've changed Birks A up and down from ND sim reco. This files now exists. The first part of the step is to get these files

## Files

### Setting the right environments and tokens and stuff

```
. /cvmfs/dune.opensciencegrid.org/spack/setup-env.sh
spack env activate dune-prototype
htgettoken -i dune -a htvaultprod.fnal.gov
```

### Grabbing the list of relevant files
```
rucio replica list file --pfns neardet-lar-tms-sand:MiniProdN5p1_NDComplex_FHC_caf_ndlarfid_sanddrift_spineonly > files_nominal.txt
rucio replica list file --pfns  neardet-lar-tms-sand:MiniProdN5p1_NDComplex_FHC_caf_ndlarfid_sanddrift_spineonly_abup > files_up.txt
rucio replica list file --pfns  neardet-lar-tms-sand:MiniProdN5p1_NDComplex_FHC_caf_ndlarfid_sanddrift_spineonly_abdown > files_down.txt
```

You can use `| head -n 5` to grab the first few events and stuff.


## RNTuple
Now that we have the list of files, we can use the converter to grab these files. 

In a different terminal:
 
### Build

```
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh
mkdir build && cd build
cmake .. && make install
```

### Convert

```
xargs -a files_nominal.txt ./converter rntuple.root
xargs -a files_up.txt ./converter rntuple_up.root
xargs -a files_down.txt ./converter rntuple_down.root
```
and et voila! you have the files!

## Making some plots

Plots are generally made from the `CAF_plotter.py` helper script. There are two classes in there, one is meant for the loading and accessing of CAFs, the other one is primarily for plotting. There is also a pretty big set of configuration of what plots we actually want to plot are. 

The overall theory behind this script is that we only plot 1D histograms of branches. If its a native branch, its very easy all you do is call that branch in the configuration. If its a derived branch, you will have to make a function that shows how you make said branches and then register it. There are plenty of examples in the script of how this works, but it might be bestr to just let AI deal with this bit. You can then plot a derived branch as you normally would. 


This works on LCG 108: `source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh`

After which its a matter of doing 

```
python CAF_plotter.py rntuple.root rntuple_up.root rntuple_down.root --output-dir plots
```

There are various flags in the plotter that wwould help in the future

## Making some root histograms

This is an extension of the previous helper script, done in `CAF_maker.py`. Its basically a wrapper around the helper script.

Its main job is to make a root file of the following form:
- histograms
    - A up
    - A nominal
    - A down
- weights
    - A up
    - A nominal
    - A down

so it makes the histograms and the weights, where the weights are defined as A_[whatever] / A_nominal. 

It will also make a plot of the histograms and the weights just to quickly see what they're doing.

Running it is as simple as 

```
python CAF_maker.py rntuple.root rntuple_up.root rntuple_down.root
```
(also using LCG 108)

Many useful flags exist where you can see, but most important of which might be `--bins` which lets you set the bin number, the `--x-data` which determines which branch you are reading, and `--bin-edges` if you want to set specific bin widths.


As suggested from charlotte, a set of bin edges might be useful:

https://github.com/DUNE/MaCh3_DUNE/blob/1185dab8324e5052c01bc97badbcbe28ccaacc64/Configs/Samples/SampleHandler_FD_Beam.yaml#L38

Which is made by

```
python CAF_maker.py rntuple.root rntuple_up.root rntuple_down.root --bin-edges 0 0.5 1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 3.25 3.5 3.75 4 5 6 10
```
