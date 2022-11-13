import numpy as np
import pandas as pd
from smt.surrogate_models import KRG


def check_independents(indepenents: list[np.ndarray]):
    if not all(x.ndim == 1 for x in indepenents) or not all(
        x.shape[0] == indepenents[0].shape[0] for x in indepenents
    ):
        raise ValueError(
            "Mismatch in size of inputs, each variable must be 1D and the same"
            f" length!, your input shapes are {[i.shape for i in indepenents]}"
        )


def lf_data(
    discangle: np.ndarray,
    airspeed: np.ndarray,
    propspeed: np.ndarray,
    base_propspeed: float,
    load_at_base_propspeed: float,
) -> pd.DataFrame:
    """This function produces some dummy data, which is supposed to looks a little
    like generic propeller data. All just with dimensional, simple inputs. We take in
    separate arrays for each dimension, and produce a pd.DataFrame. This is because we
    commonly work with data from logs, which can be read as a DataFrame.

    Args:
        discangle (np.ndarray): dim1, 1d array, in degrees!
        airspeed (np.ndarray): dim2, 1d array
        propspeed (np.ndarray): dim3, 1d array
        base_propspeed (float): baseline propspeed, where the load is defined
        load (float): the load coefficient value (mean)

    Returns:
        pd.DataFrame : corresponding load array
    """
    discangle = np.asarray(discangle)
    airspeed = np.asarray(airspeed)
    propspeed = np.asarray(propspeed)
    indepenents = [discangle, airspeed, propspeed]
    check_independents(indepenents)

    # coarse load calculation.
    load = (
        load_at_base_propspeed
        * (propspeed / base_propspeed) ** 2
        * (
            1
            + (
                (np.max(propspeed) ** 2 / propspeed**2)
                * (np.sin(np.deg2rad(discangle)))
                * airspeed
                / 30.0
            )
        )
    )
    return pd.DataFrame(
        {
            "airspeed": airspeed,
            "discangle": discangle,
            "propspeed": propspeed,
            "load": load,
        }
    )


def hf_data(
    discangle: np.ndarray,
    airspeed: np.ndarray,
    propspeed: np.ndarray,
    base_propspeed: float,
    load_at_base_propspeed: float,
) -> pd.DataFrame:
    """This function produces some dummy data, which is supposed to looks a little
    like generic propeller data. All just with dimensional, simple inputs. We take in
    separate arrays for each dimension, and produce a pd.DataFrame. This is because we
    commonly work with data from logs, which can be read as a DataFrame.

    Args:
        discangle (np.ndarray): dim1, 1d array, in degrees!
        airspeed (np.ndarray): dim2, 1d array
        propspeed (np.ndarray): dim3, 1d array
        base_propspeed (float): baseline propspeed, where the load is defined
        load (float): the load coefficient value (mean)

    Returns:
        pd.DataFrame : corresponding load array
    """
    discangle = np.asarray(discangle)
    airspeed = np.asarray(airspeed)
    propspeed = np.asarray(propspeed)
    indepenents = [discangle, airspeed, propspeed]
    check_independents(indepenents)
    # coarse load calculation.
    load = (
        load_at_base_propspeed
        * (propspeed / base_propspeed) ** 2
        * (
            0.95
            + (
                (np.max(propspeed) ** 3 / propspeed**3)
                * (np.sin(np.deg2rad(discangle)))
                * airspeed
                / 35.0
            )
        )
    )
    return pd.DataFrame(
        {
            "airspeed": airspeed,
            "discangle": discangle,
            "propspeed": propspeed,
            "load": load,
        }
    )
