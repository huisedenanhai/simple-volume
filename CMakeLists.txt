cmake_minimum_required(VERSION 3.15)
project(SimpleVolume LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 75)
# suppress warning from nanovdb
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w -Xcompiler /utf-8")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")

add_executable(volume 
  main.cpp 
  render.cu 
  render.h
  random.h
  defs.h
  math_consts.h
  vec_math.h)

target_compile_features(volume PRIVATE cxx_std_17)
set_target_properties(volume 
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET volume PROPERTY CUDA_ARCHITECTURES 75)

configure_file(config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)

target_include_directories(volume PRIVATE 
  ${CMAKE_CURRENT_BINARY_DIR}
  third_party/nanovdb/include
  third_party/openvdb/include
  third_party/openexr/include
  third_party/boost/include
  third_party/tbb/include
  third_party/cpptoml
)

target_compile_definitions(volume PRIVATE NOMINMAX)

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

add_subdirectory(third_party/stb)
target_link_libraries(volume PRIVATE stb)

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

add_executable(mse mse.cpp)
target_link_libraries(mse PRIVATE stb)