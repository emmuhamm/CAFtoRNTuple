#include "duneanaobj/StandardRecord/StandardRecord.h"

#include <ROOT/RNTupleReader.hxx>
#include "TH1D.h"
#include <TFile.h>


#include <chrono>
#include <iostream>

int main(int argc, char const *argv[]) {

  auto reader = ROOT::RNTupleReader::Open("caf", argv[1]);
  auto srView = reader->GetView<caf::StandardRecord>("rec");

  TH1D *muonE = new TH1D("muonE", "Muon Energy", 100, 0, 10);

  const auto loopStart = std::chrono::steady_clock::now();
  for (auto entryId : reader->GetEntryRange()) {
    const auto sr = srView(entryId);
    for (const auto &dlp : sr.common.ixn.dlp) {
      for (const auto &part : dlp.part.dlp) {
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

  TFile outFile("muonE_rntuple.root", "RECREATE");
  muonE->Write();
  outFile.Close();

  return 0;
}