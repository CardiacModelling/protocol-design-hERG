# Protocol designs for hERG

This repo contains code to reproduce the work in "[_A range of voltage-clamp protocol designs for rapid capture of hERG kinetics_](https://doi.org/10.12688/wellcomeopenres.23319.2)" by Chon Lok Lei, Dominic G. Whittaker, Monique J. Windley, Matthew D. Perry, Adam P. Hill, Gary R. Mirams.

If you use it please cite this paper:
```
Lei CL, Whittaker DJ, Windley MJ, Perry MD, Hill AP, Mirams GR.  
A range of voltage-clamp protocol designs for rapid capture of hERG kinetics.  
Wellcome Open Res 2025, 9:673  
(https://doi.org/10.12688/wellcomeopenres.23319.2)
```

All protocol designs are provided as time series CSV files in [`protocol-time-series`](protocol-time-series).

To run the code, run `pip install -r requirements.txt` to install all the necessary dependencies. Python >3.6 is required.

## Content

- `optimise-bruteforce`: Code to design `hhbrute3gstep` and `wangbrute3gstep`.
- `optimise-localsensitivity`: Code to design `hh3step` and `wang3step`.
- `optimise-sobol`: Code to design `hhsobol3step` and `wangsobol3step`.
- `optimise-square-wave`: Code to design `maxdiff`.
- `rank-spacefill-protocols`: Code to rank `spacefill26`, `spacefill10` and `spacefill19`.
- `newton`: Code to perform first round fitting with `staircase`, `sis`, `hh3step` and `wang3step`.
- `lib`: Utility modules.
- `mmt-model-files`: Model files for hERG.
