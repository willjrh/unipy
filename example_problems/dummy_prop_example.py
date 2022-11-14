import numpy as np
import pandas as pd
from random import uniform
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


def add_noise_adv_rat(
    df: pd.DataFrame,
    axial_noise_adv_rat: float,
    radius: float = 1.6,
    noise_shift: float = 1.0,
) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        axial_noise_adv_rat (float): _description_
        noise_mag (float): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # axial advance ratio: (airspeed sin(discangle)) / (omega * r)
    adv_rat: np.ndarray = (
        np.sin(np.deg2rad(df.discangle.to_numpy())) * df.airspeed.to_numpy()
    ) / (df.propspeed.to_numpy() * np.pi * radius / 30)
    # use logical indexing to add in the noise
    no_noise_load = df.load.to_numpy()
    # noise = np.random.normal(0.0, 1, size=adv_rat.shape) * 1000
    np.random.seed(0)
    noise = np.random.rand(len(adv_rat)) + 0.5
    df["load_noise"] = np.where(
        adv_rat > axial_noise_adv_rat,
        noise_shift * no_noise_load * noise,
        no_noise_load,
    )
    return df
