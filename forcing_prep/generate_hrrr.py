'''
Process HRRR forcings into timeseries for CAMELS basins & subcatchments.

    Entrypoint for resampling zarr based HRRR to hy_features catchments.

    Saves to file the following outputs:
    - Individual subcatchment forcing timeseries saved as f'{out_dir}/{year_str}/camels_{basin_id}_{year_str}/cat-{subcatchment_id}}.csv'
        where year_str = {year_begin}_to_{year_end}, e.g. '1979_to_2023'
    - Aggregated basin forcing timeseries saved as f'{out_dir}/{year_str}/camels_{basin_id}_{year_str}/{basin_id}_{year_str}_agg.csv'
    - Basin AORC coverage weightings saved as f'{out_dir}/{year_str}/{basin_id}_{year_str}_coverage.parquet'

    Record of missing forecast data through 2020 here: 
    https://mesowest.utah.edu/html/hrrr/zarr_documentation/html/fcst_downtime.html


    See Also
    --------
    generate.py for processing AORC data

    Authors
    -------
    Guy Litt <glitt@lynker.com>
    Nels Frazier <nfrazier@lynker.com>

    Version
    -------
    0.2

    Example
    -------
    python /path/to/git/CIROH_DL_NextGen/forcing_prep/generate_hrrr.py "/path/to/git/CIROH_DL_NextGen/forcing_prep/config_hrrr.yaml"

    Changelog / Contributions
    -------------------------
    2024-06-20: (v0.1) Adapted AORC processing to HRRR processing, GL
    2024-09-18: (v0.2) Add local gpkg processing, GL


'''
import argparse
import yaml
from multiprocessing.pool import ThreadPool
from pathlib import Path

import dask
import dask.delayed
import geopandas as gpd
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from dask.diagnostics import ProgressBar
import warnings

# The custom functions
from hrrr_proc import prep_date_time_range, _map_open_files_hrrrzarr, _gen_hrrr_zarr_urls
from geo_proc import process_geo_data

dask.config.set(pool=ThreadPool(12))
from functools import partial
from cartopy import crs as ccrs

def _preprocess_sel_time(xda, apcp_fcst):
    # This helps select the forecast hour of interest, rather than grab all forecasted hours
    # reference: how to pass arguments in preprocess:  https://github.com/pydata/xarray/pull/6825
    xda = xda.isel(time=apcp_fcst)
    return xda

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process the YAML config file.')
    parser.add_argument('config_path', type=str, help='Path to the YAML configuration file')
    args = parser.parse_args()
    
    # Load the YAML configuration file
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    home_dir = Path.home()

    cvar = config['cvar']
    ctime_max = config['ctime_max']
    cid = config['cid']
    redo = config['redo']
    x_lon_dim = config['x_lon_dim']
    y_lat_dim = config['y_lat_dim']
    out_dir = Path(config['out_dir'].format(home_dir=home_dir)) # out_dir = f'{Path.home()}/noaa/data/hrrr/redo'
    dir_custom_gpkg = Path(config.get('dir_custom_gpkg', '').format(home_dir=home_dir)) if config.get('dir_custom_gpkg', None) is not None else None
    epsg = config.get('epsg',None)
    id_col = config.get('id_col', 'divide_id') # Default to 'divide_id' in the case of hydrofabric


    time_bgn = config['time_bgn']# '2018-07-13'
    time_end = config['time_end']
    _bucket_subf = config['hrrr_source']
    _basin_url = config['basin_url_template']
    basins = config['basins']
    _level_vars_anl = config['level_vars_anl']
    _level_vars_fcst = config['level_vars_fcst']
    apcp_fcst_hr = config['fcst_hr'] # when the 'nowcast' is desired, this should be 0
    _drop_vars = config['drop_vars']
    
    

    actual_fcst_dt_hr = apcp_fcst_hr + 1 # for accumulated precip, the actual forecast timestamp is accumulated precip at the end of an hour, so add 1 hour. E.g. if nowcast is desired, apcp_fcst_hr = 0, but we need to add 1 hour to represent the accumulated precip that actually happened.
    ####
    fs = s3fs.S3FileSystem(anon=True)
    # List all the basins inside the hydrofabric s3 bucket path
    if 'all' in basins:
        # Expected format: 's3://lynker-spatial/hydrofabric/v20.1/camels/Gage_{basin_id}.gpkg'
        # base_path = 's3://lynker-spatial/hydrofabric/v20.1/camels/'
        base_path = str(Path(_basin_url).parent)
        if 's3://' not in base_path:
            base_path = str(base_path).replace('s3:/','s3://')
        # TODO provide an option if dir_custom_gpkg exists
        basins = np.unique([str(Path(x).stem.split('_')[1]) for x in  fs.ls(base_path) if '/Gage_' in x])

    Path.mkdir(Path(out_dir), exist_ok = True)

    # Define the partial function used for processing time in forecast data:
    partial_func = partial(_preprocess_sel_time, apcp_fcst = apcp_fcst_hr)

    all_dates, all_hours = prep_date_time_range(time_bgn, time_end)
    
    # HRRR grid uses the Lambert Conformal projection:
    proj = ccrs.LambertConformal(central_longitude=262.5, 
                                    central_latitude=38.5, 
                                    standard_parallels=(38.5, 38.5),
                                        globe=ccrs.Globe(semimajor_axis=6371229,
                                                        semiminor_axis=6371229))
    for date in all_dates:
        print(f'Processing {date}')
        try:
            urls_fcst, urls_anl =  _gen_hrrr_zarr_urls(date=date, level_vars_anl=_level_vars_anl, level_vars_fcst=_level_vars_fcst,fcst_hr=apcp_fcst_hr, bucket_subf = _bucket_subf)
        except:
            raise ValueError(f'Could not list bucket for {date} inside {_bucket_subf}.\nConsider sf.ls() in lieu of explicit build.')

        skip_fcst = skip_anl = False
        if len(urls_fcst) == 0 == len(urls_anl) == 0:
            print(f'No data exist for {date}') 
            continue
        elif len(urls_fcst) == 0:
            print(f'No forecasted precip data available on {date}')
            skip_fcst = True
        elif len(urls_anl) == 0:
            print(f'No analysis data available on {date}')
            skip_anl = True
        elif len(urls_fcst[0]) == 0:
            raise Warning(f'No forecast urls exist for {date}') # e.g. '20180711'

        # Now run a data pull
        try:
            if not skip_anl:
                dat_anl = _map_open_files_hrrrzarr(urls_ls = urls_anl, concat_dim = ['time',None])
            else: 
                dat_anl = xr.Dataset()
            if not skip_fcst:
                dat_fcst = _map_open_files_hrrrzarr(urls_ls = urls_fcst, concat_dim = ['time',None], preprocess = partial_func,fcst_hr=actual_fcst_dt_hr)
            else:
                dat_fcst = xr.Dataset()
        except: # Example: 20190506
            print(f'Initial hrrrzarr file opening unsuccessful on {date}. Waiting 30s and reattempting:') 
            import time
            time.sleep(30) # wait 30 seconds and try again
            try:
                if not skip_anl:
                    dat_anl = _map_open_files_hrrrzarr(urls_ls = urls_anl, concat_dim = ['time',None])
                else: 
                    dat_anl = xr.Dataset()
                if not skip_fcst:
                    dat_fcst = _map_open_files_hrrrzarr(urls_ls = urls_fcst, concat_dim = ['time',None], preprocess = partial_func,fcst_hr=actual_fcst_dt_hr)
                else:
                    dat_fcst = xr.Dataset()
            except:
                raise ValueError(f'TODO figure out what to do for {date}') 

        dat_anl = dat_anl.drop_vars([x for x in dat_anl.data_vars.keys() if x in _drop_vars])
        dat_fcst = dat_fcst.drop_vars([x for x in dat_fcst.data_vars.keys() if x in _drop_vars])
        forcing = dat_anl.merge(dat_fcst)   
        
        for b in basins:
            print(f'Processing basin {b}')
            if not dir_custom_gpkg: # read the geopackage from s3
                print(f"Reading geopackage data from s3: {_basin_url}")
                gdf = gpd.read_file(
                    fs.open(_basin_url.format(b)), driver="gpkg", layer="divides").to_crs(proj)
            else: # read the geopackage locally
                all_files = list(dir_custom_gpkg.glob('*.gpkg'))
                
                gpkg_file = [f for f in all_files if str(b) in f.stem]
                print(f"Reading geopackage data locally from: {gpkg_file}")
                gdf_raw =gpd.read_file(gpkg_file[0],engine='pyogrio')
                if epsg:
                    gdf_raw = gdf_raw.set_crs(epsg=epsg,allow_override=True)
                else:
                    warnings.warn("EPSG NOT SPECIFIED FOR INPUT DATA!!!")
                # Convert to the grid's native projection of LambertConformal:
                # https://mesowest.utah.edu/html/hrrr/zarr_documentation/html/ex_python_plot_zarr.html#:~:text=Plotting%20HRRR%20Zarr%20data%20for%20a%20single%20gridpoint.%20This%20python
                gdf = gdf_raw.to_crs(proj)

            df = process_geo_data(gdf, data=forcing, name = b, y_lat_dim = y_lat_dim, x_lon_dim = x_lon_dim, id_col=id_col, out_dir = out_dir, redo = redo)
            df = df.to_dataframe()
            # Save results by basin average and subcatchment
            save_path_base = f'{out_dir}/camels_{date}' # Main directory based on date
            cats = df.groupby('divide_id') # Note that 'divide_id' has become a standardized colname at this point
            path = Path(save_path_base)
            Path.mkdir(path, exist_ok=True)
            for name, data in cats:
                data = data.droplevel('divide_id')
                data.to_csv(path / f"{name}.csv")
            agg = df.groupby("time").mean()
            agg.to_csv(path / f"camels_{b}_agg.csv")