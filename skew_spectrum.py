from __future__ import print_function, division

from collections import OrderedDict
import numpy as np
from nbodykit.source.mesh.field import FieldMesh

from lsstools.nbkit03_utils import get_cstats_string, calc_quadratic_field

class LinField(object):
    def __init__(self, name=None, n=0, m=None, prefactor=1.0):
        self.name = name
        self.n = n
        self.m = m
        if self.m is None:
            self.m = [0,0,0]
        self.prefactor = prefactor
        if self.name is None:
            self.name = 'n%d_m%d%d%d' % (
                self.n, self.m[0], self.m[1], self.m[2])
            if self.prefactor != 1.0:
                self.name = '%g %s' % (self.prefactor, self.name)

    def compute_from_mesh(self, mesh, mode='real'):
        dnm = compute_dnm(mesh, n=self.n, m=self.m)
        if self.prefactor != 1.0:
            linmesh = FieldMesh(self.prefactor*dnm.compute(mode=mode))
        else:
            linmesh = FieldMesh(dnm.compute(mode=mode))
        return linmesh

    def to_dict(self):
        return dict(
            name=self.name,
            n=self.n,
            m=self.m,
            prefactor=self.prefactor
            )


class QuadField(object):
    def __init__(self, name=None, 
        n=0, m=None, nprime=0, mprime=None,
        nprimeprime=0, mprimeprime=None,
        prefactor=1.0,
        composite=None):
        """
        Object to represent k^n'' \vk^m'' [d_nm(x) * d_n'm'(x)].
        """

        self.name = name
        self.n = n
        self.m = m
        self.nprime = nprime
        self.mprime = mprime
        self.nprimeprime = nprimeprime
        self.mprimeprime = mprimeprime

        if self.m is None:
            self.m = [0,0,0]
        if self.mprime is None:
            self.mprime = [0,0,0]
        if self.mprimeprime is None:
            self.mprimeprime = [0,0,0]
        self.composite = composite
        self.prefactor = prefactor
        
        if self.name is None:
            if self.composite is None:
                self.name = ''
            else:
                self.name = self.composite
            self.name += 'n%d_m%d%d%d_nprime%d_mprime%d%d%d_nprimeprime%d_mprimeprime%d%d%d' % (
                self.n, self.m[0], self.m[1], self.m[2],
                 self.nprime, self.mprime[0], self.mprime[1], self.mprime[2],
                 self.nprimeprime, self.mprimeprime[0], self.mprimeprime[1], self.mprimeprime[2])
            if self.prefactor != 1.0:
                self.name = '%g %s' % (self.prefactor, self.name) 

    def to_dict(self):
        return dict(
            name=self.name,
            n=self.n,
            m=self.m,
            nprime=self.nprime,
            mprime=self.mprime,
            nprimeprime=self.nprimeprime,
            mprimeprime=self.mprimeprime,
            composite=self.composite,
            prefactor=self.prefactor
            )


    def compute_from_mesh(self, mesh, second_mesh=None, mode='real'):
        if self.composite is None:
            quadmesh = compute_dnm_dnmprime(
                mesh, mesh_prime=second_mesh, 
                n=self.n, nprime=self.nprime, m=self.m, mprime=self.mprime
                )

        else:
            dnm = compute_dnm(mesh, n=self.n, m=self.m)

            if self.composite in ['F2', 'velocity_G2', 'tidal_G2']:
                if (self.n==self.nprime) and np.all(self.m==self.mprime):
                    # two meshs are equal
                    quadmesh = calc_quadratic_field(
                        base_field_mesh=dnm, 
                        quadfield=self.composite)
                else:
                    # two different meshs
                    dnmprime = compute_dnm(second_mesh, n=self.nprime, m=self.mprime)
                    quadmesh = calc_quadratic_field(
                        base_field_mesh=dnm, 
                        second_base_field_mesh=dnmprime,
                        quadfield=self.composite+'_two_meshs')

            elif self.composite == 'velocity_G2_par_LOS001':
                dnmprime = compute_dnm(second_mesh, n=self.nprime, m=self.mprime)
                quadmesh = calc_quadratic_field(
                    base_field_mesh=dnm, 
                    second_base_field_mesh=dnmprime,
                    quadfield=self.composite+'_two_meshs')

            else:
                raise Exception('Invalid composite %s' % str(self.composite))

        if (self.nprimeprime!=0) or (not np.all(self.mprimeprime==[0,0,0])):
            # apply external derivative
            quadmesh = compute_dnm(FieldMesh(quadmesh.compute(mode='complex')), 
                n=self.nprimeprime, 
                m=self.mprimeprime)

        if self.prefactor != 1.0:
            quadmesh = FieldMesh(self.prefactor*quadmesh.compute(mode=mode))

        return quadmesh


class SumOfQuadFields(object):
    def __init__(self, quad_fields=None, name=None):
        self.quad_fields = quad_fields
        if self.quad_fields is None:
            self.quad_fields = []
        self.name = name
        if self.name is None:
            self.name = ' + '.join([qf.name for qf in self.quad_fields])

    def compute_from_mesh(self, mesh, second_mesh=None,
        mode='real'):
        out_rfield = None
        for quad_field in self.quad_fields:
            if out_rfield is None:
                out_rfield = quad_field.compute_from_mesh(
                    mesh=mesh, second_mesh=second_mesh,
                    mode=mode).compute(mode=mode)
            else:
                out_rfield += quad_field.compute_from_mesh(
                    mesh=mesh, second_mesh=second_mesh,
                    mode=mode).compute(mode=mode)
        return FieldMesh(out_rfield)

    def to_dict(self):
        d = dict()
        for counter, quad_field in enumerate(self.quad_fields):
            d['quad_field_%d' % counter] = quad_field.to_dict()
        return d


class SkewSpectrumV2(object):
    """
    SkewSpectrum class, version 2.
    Uses SumOfQuadFields and QuadFields classes.
    Compute <quad_field, delta> or <sum_of_quad_fields, delta>
    """
    def __init__(self, quad, lin=None, name=None, LOS=None):
        """
        Parameters
        ----------
        lin : LinField object
        quad : QuadField or SumOfQuadFields object
        """
        self.name = name
        self.lin = lin
        self.quad = quad
        self.LOS = LOS
        self.Pskew = None

        if self.lin is None:
            self.lin = LinField()

        if self.name is None:
            self.name = '%s_X_%s' % (self.quad.name, self.lin.name)


    def compute_from_mesh(self, mesh, second_mesh=None, third_mesh=None, 
                          power_kwargs={'mode': '2d', 'poles':[0]}):
        """
        Compute <quadfield[mesh, second_mesh], linfield[third_mesh]>.
        """
        if second_mesh is None:
            second_mesh = mesh
        if third_mesh is None:
            third_mesh = mesh

        quadratic_mesh = self.quad.compute_from_mesh(
            mesh, second_mesh=second_mesh)
        
        lin_mesh = self.lin.compute_from_mesh(third_mesh)

        self.Pskew = calc_power(quadratic_mesh, second=lin_mesh,
            los=self.LOS, **power_kwargs)

        return self.Pskew

    def to_dict(self):
        return dict(
            name=self.name,
            lin=self.lin.to_dict(),
            quad=self.quad.to_dict(),
            LOS=self.LOS,
            Pskew=self.Pskew.__getstate__()
            )

    def save_plaintext(self, filename):
        poles = self.Pskew.attrs['poles']
        d = OrderedDict()
        d['k'] = self.Pskew.poles['k']
        for ell in poles:
            d['power_%d'%ell] = self.Pskew.poles['power_%d'%ell].real

        # convert to structured numpy array
        mydtype = []
        for k in d.keys():
            mydtype.append((k,'f8'))
        arr = np.empty(d['k'].shape, dtype=mydtype)
        for k, v in d.items():
            arr[k] = v

        header = 'lin: ' + str(self.lin.to_dict()) + '\n'
        header += 'quad: ' + str(self.quad.to_dict()) + '\n'
        header += 'Pskew attrs: ' + str(self.Pskew.attrs) + '\n\n'
        header += 'Columns: ' + str(arr.dtype.names)
        np.savetxt(filename, arr, header=header)
        print('Wrote %s' % filename)

    # def to_json(self, filename=None):
    #     # Save nbodykit's BinnedStatistic
    #     #import json
    #     #from nbodykit.utils import JSONEncoder
    #     #state = self.to_dict()
    #     #with open(filename, 'w') as ff:
    #     #    json.dump(state, ff, cls=JSONEncoder)
    #     self.Pskew.to_json(filename=filename)


    # @classmethod
    # def from_json(cls, filename=None):
    #     from nbodykit.binned_statistic import BinnedStatistic
    #     obj = cls(lin=None, quad=None, LOS=None, name=None)
    #     obj.Pskew = BinnedStatistic.from_json(filename)
    #     return obj
        


class SkewSpectrum(object):
    """
    SkewSpectrum class, version 1, depracted -- don't use!
    """
    def __init__(self, name=None, smoothing=None,
                n=0, nprime=0, nprimeprime=0, m=None, mprime=None, mprimeprime=None,
                 prefactor=1.0,
                LOS=None):
        self.name = name
        self.smoothing = smoothing
        self.n = n
        self.nprime = nprime
        self.nprimeprime = nprimeprime
        if m is None:
            m = [0,0,0]
        if mprime is None:
            mprime = [0,0,0]
        if mprimeprime is None:
            mprimeprime = [0,0,0]
        self.m = m
        self.mprime = mprime
        self.mprimeprime = mprimeprime
        self.LOS = LOS
        self.Pskew = None
        self.prefactor = prefactor

    def compute_from_mesh(self, mesh, second_mesh=None, third_mesh=None, 
                          power_kwargs={'mode': '2d', 'poles':[0,2]}):
        if second_mesh is None:
            second_mesh = mesh
        if third_mesh is None:
            third_mesh = mesh
        
        quadratic_mesh = compute_dnm_dnmprime(
            mesh, mesh_prime=second_mesh, 
            n=self.n, nprime=self.nprime, m=self.m, mprime=self.mprime)
        
        linear_mesh = compute_dnm(third_mesh, n=self.nprimeprime, m=self.mprimeprime, 
                                        prefactor=self.prefactor)
        
        self.Pskew = calc_power(quadratic_mesh, second=linear_mesh, los=self.LOS, **power_kwargs)
        return self.Pskew
        
        
def compute_dnm(mesh, n, m, prefactor=1.0, verbose=True, mode='real'):
    
    # last case is most general. define simpler cases separately for speedup.
    if m[0]==0 and m[1]==0 and m[2]==0:
        if n == 0:
            def filter_fcn(k3vec, val, n=n, m=m):
                return prefactor * val
        else:
            def filter_fcn(k3vec, val, n=n, m=m):
                kk = np.sqrt(sum(ki**2 for ki in k3vec))  # |k| on the mesh
                if n<0:
                    kk[kk == 0] = 1
                return prefactor * kk**n * val

    elif m[0]==0 and m[1]==0 and m[2]!=0:
        def filter_fcn(k3vec, val, n=n, m=m):
            kk = np.sqrt(sum(ki**2 for ki in k3vec))  # |k| on the mesh
            if n<0:
                kk[kk == 0] = 1
            return prefactor * kk**n * k3vec[2]**(m[2]) * val
    
    else:
        def filter_fcn(k3vec, val, n=n, m=m):
            kk = np.sqrt(sum(ki**2 for ki in k3vec))  # |k| on the mesh
            if n<0:
                kk[kk == 0] = 1
            return prefactor * kk**n * k3vec[0]**(m[0]) * k3vec[1]**(m[1]) * k3vec[2]**(m[2]) * val

    out_mesh = FieldMesh(mesh.compute(mode='complex'))
    dnm = out_mesh.apply(filter_fcn, mode='complex', kind='wavenumber').compute(mode=mode)
    del filter_fcn

    if verbose:
        print('d_%d^%d%d%d: ' % (n, m[0], m[1], m[2]), get_cstats_string(dnm))

    return FieldMesh(dnm)

def compute_dnm_dnmprime(mesh, mesh_prime=None,
                     n=None, nprime=None, m=None, mprime=None, verbose=True,
                     ):
    """
    Parameters
    ----------
    n : int
    nprime : int
    m : (3,) sequence 
    mprime : (3,) sequence 
    """
    dnm_x = compute_dnm(mesh, n, m, verbose=True).compute(mode='real')
    if mesh_prime is None:
        mesh_prime = mesh
    dnm_x_prime = compute_dnm(mesh_prime, nprime, mprime, verbose=True).compute(mode='real')

    out_rfield = dnm_x * dnm_x_prime

    if verbose:
        print('d_%d^%d%d%d: ' % (n, m[0], m[1], m[2]), get_cstats_string(dnm_x))
        print('d_%d^%d%d%d prime: ' % (nprime, mprime[0], mprime[1], mprime[2]), get_cstats_string(dnm_x_prime))
        print('d*dprime: ', get_cstats_string(out_rfield))


    return FieldMesh(out_rfield)




from nbodykit.lab import FFTPower, FieldMesh
def calc_power(mesh, second=None, mode='1d', k_bin_width=1.0, verbose=False, los=None, poles=None):
    BoxSize = mesh.attrs['BoxSize']
    assert BoxSize[0] == BoxSize[1]
    assert BoxSize[0] == BoxSize[2]
    boxsize = BoxSize[0]
    dk = 2.0 * np.pi / boxsize * k_bin_width
    kmin = 2.0 * np.pi / boxsize / 2.0

    if mode == '1d':
        res = FFTPower(first=mesh,
                        second=second,
                        mode=mode,
                        dk=dk,
                        kmin=kmin)
    elif mode == '2d':
        if poles is None:
            poles = [0,2,4]
        res = FFTPower(first=mesh,
                            second=second,
                            mode=mode,
                            dk=dk,
                            kmin=kmin,
                            poles=poles,
                            Nmu=5,
                            los=los)
    else:
        raise Exception("Mode not implemented: %s" % mode)

    return res

