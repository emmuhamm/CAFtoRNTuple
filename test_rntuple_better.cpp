#include "duneanaobj/StandardRecord/StandardRecord.h"

#include <ROOT/RNTupleReader.hxx>
#include "TH1D.h"
#include <TFile.h>


#include <chrono>
#include <iostream>

int main(int argc, char const *argv[]) {

  auto reader = ROOT::RNTupleReader::Open("caf", argv[1]);
  auto commonView = reader->GetView<caf::SRCommonRecoBranch>("rec.common");

  TH1D *muonE = new TH1D("muonE", "Muon Energy", 100, 0, 10);

  const auto loopStart = std::chrono::steady_clock::now();

  for (auto i : reader->GetEntryRange()) {
    const auto common = commonView(i);
    for (const auto &ixn : common.ixn.dlp) {
      for (const auto &p : ixn.part.dlp) {
        if (p.pdg == 13 || p.pdg == -13) { // muon
        muonE->Fill(p.E);
        }
      }
    }
  }

  // for (auto i : reader->GetEntryRange()) {
  //   const auto parts = partsView(i);
  //   for (const auto &p : parts) {
  //     if (p.pdg == 13 || p.pdg == -13) { // muon
  //         muonE->Fill(p.E);
  //     }
  //   }
  // }
  

  const auto loopEnd = std::chrono::steady_clock::now();
  const std::chrono::duration<double> loopDuration = loopEnd - loopStart;
  std::cout << "Event-processing loop execution time: "
            << loopDuration.count() << " s" << std::endl;

  TFile outFile("muonE_rntuple_better.root", "RECREATE");
  muonE->Write();
  outFile.Close();

  return 0;
}