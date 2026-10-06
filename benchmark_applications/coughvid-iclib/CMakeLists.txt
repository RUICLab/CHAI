cmake_minimum_required(VERSION 3.0)

FOREACH(METHOD "MS" "AS" "QR")
  set(TESTNAME "coughvid-iclib-${METHOD}-${TARGET_ARCH}")
  add_executable( ${TESTNAME}
    coughvid.c
  )

  target_compile_options(${TESTNAME} PRIVATE -Os)

  include(${CMAKE_CURRENT_LIST_DIR}/../cmake/tail.cmake)
ENDFOREACH()
