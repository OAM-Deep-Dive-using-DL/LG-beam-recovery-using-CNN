from turbulence import sweep_cn2_for_modes, plot_mode_overlap_vs_cn2

modes = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 1)]
cn2_values = [1e-16, 3e-16, 1e-15, 3e-15, 1e-14]

sweep_results = sweep_cn2_for_modes(
    modes=modes,
    cn2_values=cn2_values,
    wavelength=1550e-9,
    w0=25e-3,
    distance=1000.0,
    L0=10.0,
    l0=0.005,
    N=256,
    oversampling=1,
    num_screens=20,
    num_ensembles=5,
    cn2_model="uniform",
)

fig = plot_mode_overlap_vs_cn2(
    sweep_results,
    modes,
    save_path="/Users/srivatsadavuluri/Developer/FSO beam recovery/models/LDPC + Pilot + MMSE trials/plots/mode_overlap_vs_cn2.png",  # or set to a PNG path under your plots dir
    log_x=True,
)