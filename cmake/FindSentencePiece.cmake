# FindSentencePiece.cmake - Locate SentencePiece library
# This module defines:
#  SENTENCEPIECE_FOUND - System has SentencePiece
#  SENTENCEPIECE_INCLUDE_DIRS - The SentencePiece include directories
#  SENTENCEPIECE_LIBRARIES - The libraries needed to use SentencePiece

# Try to find SentencePiece via pkg-config first
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_SENTENCEPIECE QUIET sentencepiece)
endif()

# Find include directory
find_path(SENTENCEPIECE_INCLUDE_DIR
    NAMES sentencepiece_processor.h
    PATHS
        ${PC_SENTENCEPIECE_INCLUDE_DIRS}
        /usr/local/include
        /opt/homebrew/include
        /usr/include
    PATH_SUFFIXES sentencepiece
)

# Find library
find_library(SENTENCEPIECE_LIBRARY
    NAMES sentencepiece libsentencepiece
    PATHS
        ${PC_SENTENCEPIECE_LIBRARY_DIRS}
        /usr/local/lib
        /opt/homebrew/lib
        /usr/lib
)

# Handle standard arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SentencePiece
    REQUIRED_VARS SENTENCEPIECE_LIBRARY SENTENCEPIECE_INCLUDE_DIR
    FAIL_MESSAGE "Could not find SentencePiece. Install via: brew install sentencepiece"
)

if(SENTENCEPIECE_FOUND)
    set(SENTENCEPIECE_INCLUDE_DIRS ${SENTENCEPIECE_INCLUDE_DIR})
    set(SENTENCEPIECE_LIBRARIES ${SENTENCEPIECE_LIBRARY})

    message(STATUS "SentencePiece include dirs: ${SENTENCEPIECE_INCLUDE_DIRS}")
    message(STATUS "SentencePiece libraries: ${SENTENCEPIECE_LIBRARIES}")

    # Create imported target
    if(NOT TARGET SentencePiece::sentencepiece)
        add_library(SentencePiece::sentencepiece UNKNOWN IMPORTED)
        set_target_properties(SentencePiece::sentencepiece PROPERTIES
            IMPORTED_LOCATION "${SENTENCEPIECE_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${SENTENCEPIECE_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(SENTENCEPIECE_INCLUDE_DIR SENTENCEPIECE_LIBRARY)
