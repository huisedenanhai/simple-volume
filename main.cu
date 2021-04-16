#include <config.h>
#include <iostream>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/OpenToNanoVDB.h>
#include <openvdb/io/Stream.h>
#include <openvdb/openvdb.h>

int main() {
  std::cout << DATA_DIR << std::endl;
  openvdb::initialize();
  try {
    std::ifstream is(DATA_DIR "/smoke.vdb", std::ios_base::binary);
    auto grids = openvdb::io::Stream(is).getGrids();
    auto handle = nanovdb::openToNanoVDB(grids->at(0));
    std::cout << "load success" << std::endl;
  } catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
  }
}