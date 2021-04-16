cmake_minimum_required(VERSION 3.15)
project(SimpleVolume LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 72)

add_executable(volume main.cu)
target_compile_features(volume PRIVATE cxx_std_14)
set_target_properties(volume 
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET volume PROPERTY CUDA_ARCHITECTURES 72)

configure_file(config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)

target_include_directories(volume PRIVATE 
  ${CMAKE_CURRENT_BINARY_DIR}
  third_party/nanovdb/include
  third_party/openvdb/include
  third_party/openexr/include
  third_party/boost/include
  third_party/tbb/include
)

target_link_directories(volume PRIVATE 
  ${PROJECT_SOURCE_DIR}/third_party/openvdb/lib
  ${PROJECT_SOURCE_DIR}/third_party/tbb/lib/$<$<CONFIG:Debug>:debug>
  ${PROJECT_SOURCE_DIR}/third_party/boost/lib
  )

target_link_libraries(volume PRIVATE 
  "$<IF:$<CONFIG:Debug>,openvdb_d,openvdb>"
  "$<IF:$<CONFIG:Debug>,tbb_debug,tbb>"
  libboost_system-vc141-mt-x64-1_70
  libboost_iostreams-vc141-mt-x64-1_70
  )

set(OPENEXR_BUILD_SHARED ON)
add_subdirectory(third_party/ilmbase)
target_link_libraries(volume PRIVATE IlmBase::Half)

macro(copy_post_build target dll)
 add_custom_command(TARGET ${target} POST_BUILD  
  COMMAND ${CMAKE_COMMAND} -E copy_if_different  
    ${dll}
    $<TARGET_FILE_DIR:${target}>)                 
endmacro()

copy_post_build(volume
      "${PROJECT_SOURCE_DIR}/third_party/openvdb/bin/$<IF:$<CONFIG:Debug>,openvdb_d,openvdb>.dll")

copy_post_build(volume
      "${PROJECT_SOURCE_DIR}/third_party/tbb/lib/$<IF:$<CONFIG:Debug>,debug/tbb_debug.dll,tbb.dll>")

copy_post_build(volume
      "$<TARGET_FILE:IlmBase::Half>")

