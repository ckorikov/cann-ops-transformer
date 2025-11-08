import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from ascendc_extension import AscendCExtension
CURRENT_DIR = os.path.dirname(__file__)

package_name = 'rope_matrix'  # pip package name
setup(
    name=package_name,
    ext_modules=[
        AscendCExtension(
            name=package_name,
            sources=['torch_interface.cpp', "op_host/rope_matrix_tiling.cpp"],   #host tiling code
            extra_include_dirs=[
                os.path.join(CURRENT_DIR, "build_kernel", "include", "rope_matrix_custom_kernel"),
                os.path.join(CURRENT_DIR, "inc")
            ],
            extra_library_dirs=[
                os.path.join(CURRENT_DIR, "build_kernel", "lib")
            ],
            extra_libraries=['rope_matrix_custom_kernel']
            ),
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
