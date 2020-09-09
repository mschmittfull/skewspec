from __future__ import print_function, division
from argparse import ArgumentParser
from collections import OrderedDict
import numpy as np
import os

from nbodykit.lab import FFTPower, FieldMesh

from lsstools.cosmo_model import CosmoModel
from lsstools.gen_cosmo_fcns import calc_f_log_growth_rate, generate_calc_Da
from lsstools.paint_utils import mass_weighted_paint_cat_to_delta
from perr.path_utils import get_in_path
from lsstools.nbkit03_utils import get_csum, get_csqsum, apply_smoothing, catalog_persist, get_cstats_string, linear_rescale_fac, get_crms,convert_nbk_cat_to_np_array
from perr_private.model_target_pair import ModelTargetPair, Model, Target
from lsstools.sim_galaxy_catalog_creator import PTChallengeGalaxiesFromRockstarHalos
from perr_private.read_utils import readout_mesh_at_cat_pos

from skewspec import smoothing
from skewspec.skew_spectrum import SkewSpectrumV2, QuadField, SumOfQuadFields, LinField, compute_dnm, compute_dnm_dnmprime





def main():
    """
    Calculate skew spectra.
    """

    #####################################
    # PARSE COMMAND LINE ARGS
    #####################################
    ap = ArgumentParser()

    ap.add_argument('--SimSeed',
                    type=int,
                    default=400,
                    help='Simulation seed to load.')

    ap.add_argument('--boxsize',
                    default=1500.0,
                    type=float,
                    help='Boxsize in Mpc/h.')

    ap.add_argument('--ApplyRSD',
                    type=int,
                    default=0,
                    help='0: No RSD. 1: Include RSD in catalog.')

    ap.add_argument('--Rsmooth',
                    default=10.0,
                    type=float,
                    help='Smoothing of quad field.')

    ap.add_argument('--Ngrid', 
                    default=32, 
                    type=int,
                    help='Ngrid used to compute skew spectra.')


    cmd_args = ap.parse_args()

    #####################################
    # OPTIONS
    #####################################
    opts = OrderedDict()

    opts['code_version_for_outputs'] = '0.1'

    # Parse options
    opts['sim_seed'] = cmd_args.SimSeed
    opts['boxsize'] = cmd_args.boxsize
    basedir = os.path.expandvars('$SCRATCH/lss/ms_gadget/run4/00000%d-01536-%.1f-wig/' % (
        opts['sim_seed'], opts['boxsize']))
    opts['sim_scale_factor'] = 0.625
    opts['Rsmooth'] = cmd_args.Rsmooth
    opts['Ngrid'] = cmd_args.Ngrid
    opts['LOS'] = np.array([0,0,1])
    opts['APPLY_RSD'] = bool(cmd_args.ApplyRSD)
    # which multipoles (ell) to compute
    opts['poles'] = [0,2]

    opts['outdir'] = '../data/Pskew_sims/00000%d-01536-%.1f-wig/R%.1f_Ng%d_RSD%d/' % (
        opts['sim_seed'], opts['boxsize'], opts['Rsmooth'], opts['Ngrid'],
        int(opts['APPLY_RSD']))

    # Catalog with particle positions: 'DM_subsample' or 'gal_ptchall_with_RSD'
    opts['positions_catalog'] = 'DM_subsample'

    #opts['velocity_source'] = 'deltalin_D2'
    opts['velocity_source'] = 'DM_sim'


    # cosmology of ms_gadget sims (to compute D_lin(z))
    # omega_m = 0.307494
    # omega_bh2 = 0.022300
    # omega_ch2 = 0.118800
    # h = math.sqrt((omega_bh2 + omega_ch2) / omega_m) = 0.6774
    opts['cosmo_params'] = dict(Om_m=0.307494,
                       Om_L=1.0 - 0.307494,
                       Om_K=0.0,
                       Om_r=0.0,
                       h0=0.6774)

    opts['f_log_growth'] = np.sqrt(0.61826)



    # Below, 'D' stands for RSD displacement in Mpc/h: D=v/(aH)=f*PsiDot.

    ## Target catalogs
    # DM subsample
    DM_subsample = Target(
        name='DM_subsample',
        in_fname=os.path.join(basedir, 'snap_%.4f_sub_sr0.0015_ssseed40%d.bigfile' % (
            opts['sim_scale_factor'], opts['sim_seed'])),
        position_column='Position'
    )

    DM_D2 = Target(
        name='DM_D2',
        in_fname=os.path.join(basedir, 'snap_%.4f_sub_sr0.0015_ssseed40%d.bigfile' % (
            opts['sim_scale_factor'], opts['sim_seed'])),
        position_column='Position',
        val_column='Velocity',
        val_component=2,
        rescale_factor='RSDFactor'
    )

    if False:
        # PT Challenge galaxies from rockstar halos. Rockstar gives core positions and velocities.
        # Units: 1/(aH) = 1./(a * H0*np.sqrt(Om_m/a**3+Om_L)) * (H0/100.) in Mpc/h / (km/s).
        # For ms_gadget, get 1/(aH) = 0.01145196 Mpc/h/(km/s) = 0.0183231*0.6250 Mpc/h/(km/s).
        # Note that MP-Gadget files have RSDFactor=1/(a^2H)=0.0183231 for a=0.6250 b/c they use a^2\dot x for Velocity.
        assert opts['sim_scale_factor'] == 0.625

        # PT challenge galaxies, apply RSD to position (TEST)
        assert opts['sim_scale_factor'] == 0.625
        gal_ptchall_with_RSD = Target(
            name='gal_ptchall_with_RSD',
            in_fname=os.path.join(basedir, 'snap_%.4f.gadget3/rockstar_out_0.list.parents.bigfile' % opts['sim_scale_factor']),
            position_column='Position',
            velocity_column='Velocity', 
            apply_RSD_to_position=True,
            RSD_los=LOS,
            RSDFactor=0.01145196,
            #val_column='Velocity', # This is rockstar velocity, which is v=a\dot x in km/s ("Velocities in km / s (physical, peculiar)")
            #val_component=0,
            #rescale_factor=0.01145196, # RSD displacement in Mpc/h is D=v/(aH)=0.01145196*v. 
            cuts=[PTChallengeGalaxiesFromRockstarHalos(
                    log10M_column='log10Mvir', log10Mmin=12.97, sigma_log10M=0.35, RSD=False),
                  #('Position', 'max', [100.,100.,20.])
                 ]
            )


    ### Models for density
    z_rescalefac = linear_rescale_fac(current_scale_factor=1.0,
                                      desired_scale_factor=opts['sim_scale_factor'],
                                      cosmo_params=opts['cosmo_params'])
    print('z_rescalefac:', z_rescalefac)
    deltalin = Model(
        name='deltalin',
        in_fname=os.path.join(basedir, 'IC_LinearMesh_z0_Ng%d' % opts['Ngrid']),
        rescale_factor=z_rescalefac,
        read_mode='delta from 1+delta',
        filters=None,
        readout_window='cic')

    # Linear RSD displacement
    def k2ovksq_filter_fcn(k, v, d=2):
        ksq = sum(ki**2 for ki in k)
        return np.where(ksq == 0.0, 0*v, 1j*k[d] * v / (ksq))
    deltalin_D2 = Model(
        name='deltalin_D2',
        in_fname=os.path.join(basedir, 'IC_LinearMesh_z0_Ng%d' % opts['Ngrid']),
        rescale_factor=opts['f_log_growth']*z_rescalefac,
        read_mode='delta from 1+delta',
        filters=[k2ovksq_filter_fcn],
        readout_window='cic')




    ##########################################################################
    # Get DM catalog in redshift space (if APPLY_RSD==True) 
    ##########################################################################

    # get the catalog
    if opts['positions_catalog'] == 'gal_ptchall_with_RSD':
        target = gal_ptchall_with_RSD
    elif opts['positions_catalog'] == 'DM_subsample':
        target = DM_subsample
    cat = target.get_catalog(keep_all_columns=True)
    
    if cat.comm.rank == 0:
        print('Positions catalog:')
        print(cat.attrs)

    # add redshift space positions, assuming LOS is in z direction
    if opts['velocity_source'] == 'DM_sim':
        # use DM velocity
        cat['RSDPosition'] = cat['Position'] + cat['Velocity']*cat.attrs['RSDFactor'] * opts['LOS']

    elif opts['velocity_source'] == 'deltalin_D2':
        raise Exception('Should not use linear velocity b/c it will not have 2nd order G2 velocity')
        assert np.all(opts['LOS'] == np.array([0,0,1]))
        mtp = ModelTargetPair(model=deltalin_D2, target=DM_subsample)
        cat['RSDPosition'] = cat['Position'].compute()
        mat = np.zeros((cat.csize,3))
        mat[:,0] = 0*mat[:,0]
        mat[:,1] = 0*mat[:,0]
        mat[:,2] = mtp.readout_model_at_target_pos()
        cat['RSDPosition'] += mat
        
    if cat.comm.rank == 0:
        print('rms RSD displacement: %g Mpc/h' % np.mean((cat['Position'].compute()-cat['RSDPosition'].compute())**2)**0.5)
        print('max RSD displacement: %g Mpc/h' % np.max(np.abs(cat['Position'].compute()-cat['RSDPosition'].compute())))
        
    # Get redshift space catalog
    RSDcat = catalog_persist(cat, columns=['ID','PID','Position','RSDPosition','Velocity', 'log10Mvir'])
    del cat
    if opts['APPLY_RSD']:
        if RSDcat.comm.rank == 0:
            print('Applying RSD')
        RSDcat['Position'] = RSDcat['RSDPosition']
    else:
        if RSDcat.comm.rank == 0:
            print('Not applying RSD')


    ##########################################################################
    # Calculate power spectrum multipoles 
    ##########################################################################
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


    ## Compute density power spectrum 
    Nmesh = opts['Ngrid']
    BoxSize = np.array([opts['boxsize'], opts['boxsize'], opts['boxsize']])
    if RSDcat.comm.rank == 0:
        print('# objects: Original: %d' % (RSDcat.csize))

    # get mesh
    delta_mesh = FieldMesh(RSDcat.to_mesh(Nmesh=Nmesh, BoxSize=BoxSize, 
                          window='cic', interlaced=False, compensated=False).compute()-1)
    #if RSDcat.comm.rank == 0:
    #    print('Mesh: ', get_cstats_string(delta_mesh.compute()))

    Pdd = calc_power(delta_mesh, los=opts['LOS'], mode='2d', poles=opts['poles'])


    ##########################################################################
    # Get all RSD skew spectra
    ##########################################################################

    # apply smoothing
    smoothers = [smoothing.GaussianSmoother(R=opts['Rsmooth'])]
    delta_mesh_smoothed = FieldMesh(delta_mesh.compute(mode='real'))
    for smoother in smoothers:
        delta_mesh_smoothed = smoother.apply_smoothing(delta_mesh_smoothed)

    #if RSDcat.comm.rank == 0:        
    #    print('delta: ', get_cstats_string(delta_mesh.compute(mode='real')))
    #    print('delta smoothed: ', get_cstats_string(delta_mesh_smoothed.compute(mode='real')))

    LOS = opts['LOS']
    LOS_string = 'LOS%d%d%d' % (LOS[0], LOS[1], LOS[2])


    # Define skew spectra. default n=n'=0 and m=m'=[0,0,0].
    s1 = SkewSpectrumV2(QuadField(composite='F2'), LOS=LOS, name='S1')
    s2 = SkewSpectrumV2(QuadField(), LOS=LOS, name='S2')
    s3 = SkewSpectrumV2(QuadField(composite='tidal_G2'), LOS=LOS, name='S3')
    s4a = SkewSpectrumV2(QuadField(nprime=-2,mprime=2*LOS), LOS=LOS, name='S4a')
    s4b = SkewSpectrumV2(QuadField(m=LOS,nprime=-2,mprime=LOS), LOS=LOS, name='S4b')
    s4split = SkewSpectrumV2(
        SumOfQuadFields(quad_fields=[
            QuadField(nprime=-2, mprime=2*LOS, prefactor=1.0), 
            QuadField(m=LOS, nprime=-2, mprime=LOS, prefactor=1.0)]),
        LOS=LOS, name='S4')
    s4swap = SkewSpectrumV2(
        quad=QuadField(m=LOS, nprime=-2, mprime=LOS, prefactor=1.0),
        LOS=LOS, name='S4')
    s4 = SkewSpectrumV2(
        quad=QuadField(nprime=-2, mprime=LOS, mprimeprime=LOS, prefactor=1.0),
        LOS=LOS, name='S4')
    # s4sep = SkewSpectrumV2(
    #     quad=QuadField(nprime=-2, mprime=LOS),
    #     lin=LinField(m=LOS),
    #     LOS=LOS, name='S4sep')

    s5 = SkewSpectrumV2(SumOfQuadFields(quad_fields=[
        QuadField(composite='F2', nprime=-2, mprime=2*LOS),
        QuadField(composite='velocity_G2_par_%s' % LOS_string)
    ]), LOS=LOS, name='S5')
    s6 = SkewSpectrumV2(QuadField(nprime=-2, mprime=2*LOS), LOS=LOS, name='S6')
    s7 = SkewSpectrumV2(QuadField(nprime=-2, mprime=2*LOS, composite='tidal_G2'), LOS=LOS, name='S7')
    s8split = SkewSpectrumV2(SumOfQuadFields(quad_fields=[
        QuadField(nprime=-4, mprime=4*LOS),
        QuadField(n=-2, m=2*LOS, nprime=-2, mprime=2*LOS, prefactor=2.0),
        QuadField(m=LOS, nprime=-4, mprime=3*LOS),
        QuadField(n=-2, m=3*LOS, nprime=-2, mprime=LOS, prefactor=2.0)]), LOS=LOS, name='S8split')
    fudge_fac = 1.0
    s8 = SkewSpectrumV2(SumOfQuadFields(quad_fields=[
        QuadField(nprime=-4, mprime=3*LOS, mprimeprime=LOS, prefactor=fudge_fac),
        QuadField(n=-2, m=LOS, nprime=-2, mprime=2*LOS, mprimeprime=LOS, prefactor=2.0*fudge_fac)]),
        LOS=LOS, name='S8')

    s9 = SkewSpectrumV2(SumOfQuadFields(quad_fields=[
        QuadField(n=-2, m=2*LOS, nprime=-2, mprime=2*LOS, composite='F2'),
        QuadField(n=-2, m=2*LOS, composite='velocity_G2_par_%s' % LOS_string, prefactor=2.0)]),
        LOS=LOS, name='S9')
    s10 = SkewSpectrumV2(QuadField(n=-2, m=2*LOS, nprime=-2, mprime=2*LOS), LOS=LOS, name='S10')
    s11 = SkewSpectrumV2(QuadField(n=-2, m=2*LOS, nprime=-2, mprime=2*LOS, composite='tidal_G2'), LOS=LOS, name='S11')
    if True:
        # correct S12
        s12 = SkewSpectrumV2(SumOfQuadFields(quad_fields=[
            QuadField(n=-4, m=4*LOS, nprime=-2, mprime=LOS, mprimeprime=LOS),
            QuadField(n=-2, m=2*LOS, nprime=-4, mprime=3*LOS, mprimeprime=LOS, prefactor=2.0)]),
            LOS=LOS, name='S12')
    else:
        # play woth S12
        s12 = SkewSpectrumV2(SumOfQuadFields(quad_fields=[
            QuadField(n=-4, m=4*LOS, nprime=-2, mprime=LOS, mprimeprime=2*LOS),
            QuadField(n=-2, m=2*LOS, nprime=-4, mprime=3*LOS, mprimeprime=2*LOS, prefactor=2.0)]),
            LOS=LOS, name='S12')
    s13 = SkewSpectrumV2(
        QuadField(n=-2, m=2*LOS, nprime=-2, mprime=2*LOS, composite='velocity_G2_par_%s' % LOS_string),
        LOS=LOS, name='S13')
    if True:
        # correct S14
        s14 = SkewSpectrumV2(
            QuadField(n=-4, m=3*LOS, nprime=-4, mprime=4*LOS, mprimeprime=LOS),
            LOS=LOS, name='S14')
    else:
        # play with S14
        s14 = SkewSpectrumV2(
            QuadField(n=-4, m=3*LOS, nprime=-4, mprime=4*LOS, mprimeprime=2*LOS),
            LOS=LOS, name='S14')


    # list of skew spectra to compute
    power_kwargs={'mode': '2d', 'poles': opts['poles']}
    skew_spectra = [s1,s2, s3, s4, s5,s6, s7, s8, s9, s10, s11, s12, s13, s14]
    #skew_spectra = [S4, S4swap, S4split, S4a, S4b, S8, S12, S14]


    # compute skew spectra
    for skew_spec in skew_spectra:
        # compute and store in skew_spec.Pskew
        skew_spec.compute_from_mesh(mesh=delta_mesh_smoothed, third_mesh=delta_mesh, power_kwargs=power_kwargs)


    # store in individual files
    if RSDcat.comm.rank == 0:
        if not os.path.exists(opts['outdir']):
            os.makedirs(opts['outdir'])

        if False:
            for skew_spec in skew_spectra:
                # store as json
                fname = os.path.join(opts['outdir'], skew_spec.name+'.json')
                skew_spec.Pskew.save(fname)
                print('Wrote %s' % fname)

                # store as plain text
                fname = os.path.join(opts['outdir'], skew_spec.name+'.txt')
                skew_spec.save_plaintext(fname)
                print('Wrote %s' % fname)
            
        # store all in one file for each multipole
        for ell in skew_spec.Pskew.attrs['poles']:
            mydtype = [('k', 'f8')]
            for skew_spec in skew_spectra:
                mydtype.append((skew_spec.name, 'f8'))
            arr = np.empty(shape=skew_spec.Pskew.poles['k'].shape, dtype=mydtype)
            arr['k'] = skew_spec.Pskew.poles['k']
            for skew_spec in skew_spectra:
                arr[skew_spec.name] = skew_spec.Pskew.poles['power_%d'%ell].real
            fname = os.path.join(opts['outdir'], 'Sn_ell%d.txt'%ell)
            header = 'Columns: ' + str(arr.dtype.names)
            np.savetxt(fname, arr, header=header)
            print('Wrote %s' % fname)




if __name__ == '__main__':
    main()
