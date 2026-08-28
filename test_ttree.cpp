#include "duneanaobj/StandardRecord/StandardRecord.h"

#include "TChain.h"
#include "TFile.h"
#include "TSystem.h"
#include "TH1D.h"

#include <chrono>
#include <iostream>

int main(int argc, char const *argv[]) {

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <caf file(s)>" << std::endl;
    return 1;
  }

  TChain ch("cafTree");

  for(int i = 1; i < argc; ++i){
    ch.Add(argv[i]);
  }

  caf::StandardRecord *SR = nullptr;
  if (!ch.GetBranch("rec")) {
    std::cerr << "Error: branch 'rec' not found in cafTree."
              << " Check input file content and loaded dictionaries." << std::endl;
    return 2;
  }
  ch.SetBranchStatus("*", 1);
  ch.SetBranchStatus("nd.sand.ixn*", 0);
  ch.SetBranchAddress("rec", &SR);

  Long64_t ents = ch.GetEntries();
  if (ents <= 0) {
    std::cerr << "Error: no entries found in cafTree." << std::endl;
    return 3;
  }

  if (ch.GetEntry(0) <= 0 || SR == nullptr) {
    std::cerr << "Error: failed to read first entry from cafTree rec branch."
              << " Dictionaries may be missing at runtime." << std::endl;
    return 4;
  }

  TH1D *muonE = new TH1D("muonE", "Muon Energy", 100, 0, 10);

  const auto loopStart = std::chrono::steady_clock::now();
  for (Long64_t i = 0; i < ents; ++i) {
    if (ch.GetEntry(i) <= 0 || SR == nullptr) {
      continue;
    }
    if (SR->common.ixn.dlp.empty()) {
      continue;
    }

    for (auto &dlp : SR->common.ixn.dlp) {
      for (auto &part : dlp.part.dlp) {
        if (part.pdg == 13 || part.pdg == -13) { // muon
          muonE->Fill(part.E);
        }
      }
    }
  }
  const auto loopEnd = std::chrono::steady_clock::now();
  const std::chrono::duration<double> loopDuration = loopEnd - loopStart;
  std::cout << "Event-processing loop execution time: "
            << loopDuration.count() << " s" << std::endl;

  TFile outFile("muonE_ttree.root", "RECREATE");
  muonE->Write();
  outFile.Close();

  return 0;
}