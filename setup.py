from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='aagt',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='aagt.ext',
            sources=[
                'aagt/extensions/extra/cloud/cloud.cpp',
                'aagt/extensions/cpu/grid_subsampling/grid_subsampling.cpp',
                'aagt/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'aagt/extensions/cpu/radius_neighbors/radius_neighbors.cpp',
                'aagt/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'aagt/extensions/pybind.cpp',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
