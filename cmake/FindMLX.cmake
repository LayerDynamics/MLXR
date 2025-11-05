# FindMLX.cmake - Locate MLX framework installation
# This module defines:
#  MLX_FOUND - System has MLX
#  MLX_INCLUDE_DIRS - The MLX include directories
#  MLX_LIBRARIES - The libraries needed to use MLX
#  MLX_VERSION - The version of MLX found

# Try to find MLX using Python
find_package(Python3 COMPONENTS Interpreter REQUIRED)

if(Python3_FOUND)
    # Get MLX installation path from Python
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import mlx; import os; print(os.path.dirname(mlx.__file__))"
        OUTPUT_VARIABLE MLX_PYTHON_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE MLX_PYTHON_RESULT
    )

    if(MLX_PYTHON_RESULT EQUAL 0 AND MLX_PYTHON_PATH)
        message(STATUS "Found MLX Python package at: ${MLX_PYTHON_PATH}")

        # Set include and library directories
        set(MLX_INCLUDE_DIR "${MLX_PYTHON_PATH}/include")
        set(MLX_LIBRARY_DIR "${MLX_PYTHON_PATH}/lib")

        # Get MLX version
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import mlx; print(mlx.__version__)"
            OUTPUT_VARIABLE MLX_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )

        # Find the MLX library
        find_library(MLX_LIBRARY
            NAMES mlx libmlx
            PATHS ${MLX_LIBRARY_DIR}
            NO_DEFAULT_PATH
        )

        # Find MLX Metal library if it exists
        find_library(MLX_METAL_LIBRARY
            NAMES mlx_metal libmlx_metal
            PATHS ${MLX_LIBRARY_DIR}
            NO_DEFAULT_PATH
        )

        # Check if include directory exists
        if(EXISTS ${MLX_INCLUDE_DIR})
            set(MLX_INCLUDE_DIRS ${MLX_INCLUDE_DIR})
        else()
            message(WARNING "MLX include directory not found at ${MLX_INCLUDE_DIR}")
            set(MLX_INCLUDE_DIRS "")
        endif()

        # Set libraries list
        set(MLX_LIBRARIES "")
        if(MLX_LIBRARY)
            list(APPEND MLX_LIBRARIES ${MLX_LIBRARY})
        endif()
        if(MLX_METAL_LIBRARY)
            list(APPEND MLX_LIBRARIES ${MLX_METAL_LIBRARY})
        endif()
    endif()
endif()

# Fallback to manual search if Python method failed
if(NOT MLX_INCLUDE_DIRS)
    # Common installation paths
    set(MLX_SEARCH_PATHS
        /opt/homebrew/lib/python3.11/site-packages/mlx
        /opt/homebrew/lib/python3.12/site-packages/mlx
        /usr/local/lib/python3.11/site-packages/mlx
        /usr/local/lib/python3.12/site-packages/mlx
        $ENV{HOME}/miniconda3/envs/mlxr/lib/python3.11/site-packages/mlx
        $ENV{HOME}/anaconda3/envs/mlxr/lib/python3.11/site-packages/mlx
        $ENV{CONDA_PREFIX}/lib/python3.11/site-packages/mlx
    )

    foreach(SEARCH_PATH ${MLX_SEARCH_PATHS})
        if(EXISTS ${SEARCH_PATH}/include)
            set(MLX_INCLUDE_DIR ${SEARCH_PATH}/include)
            set(MLX_LIBRARY_DIR ${SEARCH_PATH}/lib)
            message(STATUS "Found MLX at: ${SEARCH_PATH}")
            break()
        endif()
    endforeach()

    if(MLX_INCLUDE_DIR)
        set(MLX_INCLUDE_DIRS ${MLX_INCLUDE_DIR})

        find_library(MLX_LIBRARY
            NAMES mlx libmlx
            PATHS ${MLX_LIBRARY_DIR}
            NO_DEFAULT_PATH
        )

        find_library(MLX_METAL_LIBRARY
            NAMES mlx_metal libmlx_metal
            PATHS ${MLX_LIBRARY_DIR}
            NO_DEFAULT_PATH
        )

        set(MLX_LIBRARIES "")
        if(MLX_LIBRARY)
            list(APPEND MLX_LIBRARIES ${MLX_LIBRARY})
        endif()
        if(MLX_METAL_LIBRARY)
            list(APPEND MLX_LIBRARIES ${MLX_METAL_LIBRARY})
        endif()
    endif()
endif()

# Handle standard arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MLX
    REQUIRED_VARS MLX_INCLUDE_DIRS
    VERSION_VAR MLX_VERSION
    FAIL_MESSAGE "Could not find MLX. Please install it via: pip install mlx"
)

if(MLX_FOUND)
    message(STATUS "MLX version: ${MLX_VERSION}")
    message(STATUS "MLX include dirs: ${MLX_INCLUDE_DIRS}")
    message(STATUS "MLX libraries: ${MLX_LIBRARIES}")

    # Create imported target
    if(NOT TARGET MLX::mlx)
        add_library(MLX::mlx INTERFACE IMPORTED)
        set_target_properties(MLX::mlx PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${MLX_INCLUDE_DIRS}"
        )
        if(MLX_LIBRARIES)
            set_target_properties(MLX::mlx PROPERTIES
                INTERFACE_LINK_LIBRARIES "${MLX_LIBRARIES}"
            )
        endif()
    endif()
endif()

mark_as_advanced(MLX_INCLUDE_DIR MLX_LIBRARY_DIR MLX_LIBRARY MLX_METAL_LIBRARY)
