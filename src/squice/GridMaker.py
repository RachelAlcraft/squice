"""
GridMaker: makes a grid or chunk of points given the secified size and samples

"""

from . import Matrix3d as d3


class GridMaker(object):
    def get_unit_grid(self, width, samples, depth_samples=1):
        offset = (samples - 1) / 2
        gap = 1
        if samples > 1:
            gap = width / (samples - 1)
        # if there is a depth, there will be fewer samples
        if depth_samples > 1:
            return self.get_unit_grid3d(width, samples, depth_samples)
        else:
            mat2 = d3.Matrix3d(samples, samples)
            for i in range(samples):
                for j in range(samples):
                    tpl = ((i - offset) * gap, (j - offset) * gap, 0)
                    mat2.add(i, j, data=tpl)
            return mat2

    def get_unit_grid3d(self, width, samples, depth_samples):
        offset = (samples - 1) / 2
        gap = width / (samples - 1)
        # if there is a depth, there will be fewer samples
        depth_offset = (depth_samples - 1) / 2
        mat3 = d3.Matrix3d(samples, samples, depth_samples)

        for i in range(samples):
            for j in range(samples):
                for k in range(depth_samples):
                    tpl = (
                        (i - offset) * gap,
                        (j - offset) * gap,
                        (k - depth_offset) * gap,
                    )
                    mat3.add(i, j, k, data=tpl)
        return mat3
