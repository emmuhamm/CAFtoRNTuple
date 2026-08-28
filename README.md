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