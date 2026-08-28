#include "duneanaobj/StandardRecord/StandardRecord.h"

#include "TChain.h"
#include "ROOT/RNTupleModel.hxx"
#include "ROOT/RNTupleWriter.hxx"

int main(int argc, char const *argv[]) {

  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <output rntuple file> <caf file(s)>" << std::endl;
    return 1;
  }

  TChain ch("cafTree");

  for(int i = 2; i < argc; ++i){
    ch.Add(argv[i]);
  }

  caf::StandardRecord *SR = nullptr;
  if (!ch.GetBranch("rec")) {
    std::cerr << "Error: branch 'rec' not found in cafTree."
              << " Check input file content and loaded dictionaries." << std::endl;
    return 2;
  }

  ch.SetBranchStatus("*", 1);
  ch.SetBranchStatus("nd.sand.ixn*", 0); // get segfault otherwise
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

  auto model = ROOT::RNTupleModel::Create();
  auto fldRec = model->MakeField<caf::StandardRecord>("rec");

  auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "caf", argv[1]);

  for (Long64_t i = 0; i < ents; ++i) {
    if (ch.GetEntry(i) <= 0 || SR == nullptr) {
      continue;
    }

    *fldRec = *SR;
    writer->Fill();
  }
  return 0;
}