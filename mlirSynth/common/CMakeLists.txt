add_library(common
  ContextManager.cc)
target_include_directories(common
  PUBLIC ${MHLO_INCLUDE_DIRS})
target_link_libraries(common
  PUBLIC ${MHLO_EXPORTED_TARGETS})