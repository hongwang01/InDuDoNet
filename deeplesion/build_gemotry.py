import odl
import numpy as np


## 640geo
class initialization:
    def __init__(self):
        self.param = {}
        self.reso = 512 / 416 * 0.03

        # image
        self.param['nx_h'] = 416
        self.param['ny_h'] = 416
        self.param['sx'] = self.param['nx_h']*self.reso
        self.param['sy'] = self.param['ny_h']*self.reso

        ## view
        self.param['startangle'] = 0
        self.param['endangle'] = 2 * np.pi

        self.param['nProj'] = 640

        ## detector
        self.param['su'] = 2*np.sqrt(self.param['sx']**2+self.param['sy']**2)
        self.param['nu_h'] = 641
        self.param['dde'] = 1075*self.reso
        self.param['dso'] = 1075*self.reso

        self.param['u_water'] = 0.192


def build_gemotry(param):
    reco_space_h = odl.uniform_discr(
        min_pt=[-param.param['sx'] / 2.0, -param.param['sy'] / 2.0],
        max_pt=[param.param['sx'] / 2.0, param.param['sy'] / 2.0], shape=[param.param['nx_h'], param.param['ny_h']],
        dtype='float32')

    angle_partition = odl.uniform_partition(param.param['startangle'], param.param['endangle'],
                                            param.param['nProj'])

    detector_partition_h = odl.uniform_partition(-(param.param['su'] / 2.0), (param.param['su'] / 2.0),
                                                 param.param['nu_h'])

    geometry_h = odl.tomo.FanBeamGeometry(angle_partition, detector_partition_h,
                                          src_radius=param.param['dso'],
                                          det_radius=param.param['dde'])

    ray_trafo_hh = odl.tomo.RayTransform(reco_space_h, geometry_h, impl='astra_cuda')
    return ray_trafo_hh
