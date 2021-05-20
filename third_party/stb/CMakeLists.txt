add_library(stb 
    stb_image.cpp
    stb_image.h
    stb_image_write.cpp
    stb_image_write.h
    )

target_include_directories(stb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})