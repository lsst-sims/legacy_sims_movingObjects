import numpy as np
import requests
import json


def time_string_api_format(dtobj):
    """Time format required for API"""
    return dtobj.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def start_sim_online(params_dict):
    """Post simulation online"""
    url = 'https://dev-asteroids.sandbox.googleapis.com/v1/ephemerides'
    headers = {'Content-Type': 'application/json'}
    re = requests.post(url=url, data=json.dumps(params_dict), headers=headers)
    re.raise_for_status()
    response_dict = re.json()
    return response_dict['ephToken']


def get_ephem_online(eph_token):
    """Grab response from sim online"""
    url = 'https://dev-asteroids.sandbox.googleapis.com/v1/ephemerides'
    url_response = url + '/' + eph_token
    cloud_out = requests.get(url_response).json()
    # Not completed
    if cloud_out['calcState'] != 'COMPLETED':
        return None
    # Completed
    return cloud_out['stkEphemeris']


def ephemeris_final_state(ephem_str):
    """Get final state in [km, km/s] out of a .e file string

    N.B. .e files are in [m, m/s]
    """
    split_ephem = ephem_str.splitlines()
    ephem_lines = []
    for line in split_ephem:
        split_line = line.split()
        if len(split_line) == 7:
            ephem_value = np.array(split_line).astype(float)
            ephem_lines.append(ephem_value)
    RV_end = ephem_lines[-1][1:7]
    return RV_end / 1000.0


def create_opm_dict(RV, epoch, mass=1000.0,
                    solar_rad_coeff=1.0, solar_rad_area=20.0,
                    drag_coeff=2.2, drag_area=20.0):
    """Create OPM dict required for simulation online

    Args:
        RV: state position velocity [km, km/s]  (numpy array)
        epoch: state time (datetime)
        mass: mass [kg]
        solar_rad_coeff: []
        solar_rad_area: [m^2]
        drag_coeff: []
        drag_area: [m^2]

    Out:
        opm:  (dict)
            opm.keys() = ['header', 'spacecraft', 'state_vector',
                          'ccsds_opm_vers', 'metadata']
    """
    opm = {}
    opm['ccsds_opm_vers'] = '2.0'
    metadata = {}
    metadata['center_name'] = 'SUN'
    metadata['comments'] = ['Cartesian']
    metadata['object_id'] = 'SatID'
    metadata['object_name'] = 'SatName'
    metadata['ref_frame'] = 'ICRF'
    metadata['time_system'] = 'UTC'
    opm['metadata'] = metadata
    spacecraft = {}
    spacecraft['solar_rad_coeff'] = solar_rad_coeff
    spacecraft['drag_area'] = drag_area
    spacecraft['mass'] = mass
    spacecraft['drag_coeff'] = drag_coeff
    spacecraft['solar_rad_area'] = solar_rad_area
    opm['spacecraft'] = spacecraft
    header = {}
    header['originator'] = 'Originator'
    header['creation_date'] = time_string_api_format(epoch)
    opm['header'] = header
    state_vector = {}
    state_vector['epoch'] = time_string_api_format(epoch)
    state_vector['x'] = RV[0]
    state_vector['y'] = RV[1]
    state_vector['z'] = RV[2]
    state_vector['xDot'] = RV[3]
    state_vector['yDot'] = RV[4]
    state_vector['zDot'] = RV[5]
    opm['state_vector'] = state_vector
    return opm


def create_sim_data_dict(RV, epoch, start_time, end_time,
                         step_duration_sec, mass=1000.0,
                         solar_rad_coeff=1.0, solar_rad_area=20.0,
                         drag_coeff=2.2, drag_area=2.0):
    """Create parameter dict required for simulation online

    Args:
        RV: state position velocity [km, km/s]  (numpy array)
        epoch: epoch of state (datetime)
        start_time: sim start time (datetime)
        end_time: sim end time (datetime)
        mass: mass [kg]
        solar_rad_coeff: []
        solar_rad_area: [m^2]
        drag_coeff: []
        drag_area: [m^2]

    Out:
        data:  (dict)
            data.keys() = ['opm', 'start_time',
                           'step_duration_sec', 'end_time']
    """
    data = {}
    data['start_time'] = time_string_api_format(start_time)
    data['end_time'] = time_string_api_format(end_time)
    data['step_duration_sec'] = step_duration_sec
    data['opm'] = create_opm_dict(RV, epoch, mass,
                                  solar_rad_coeff, solar_rad_area,
                                  drag_coeff, drag_area)
    return data
