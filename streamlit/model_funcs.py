import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import jax
import jax.numpy as np
from jax.random import PRNGKey
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

import covid
import covid.util as util
import covid.models.SEIRD_variable_detection
import covid.models.SEIRD_incident

data = util.load_data()

def train_model(start_date, end_date, location, warmup=0, samples=200):
    model_type = covid.models.SEIRD_incident.SEIRD

    util.run_place(data,
                   location,
                   start=start_date,
                   end=end_date,
                   model_type=model_type,
                   rw_scale=1e-1,
                   num_warmup=warmup,
                   num_samples=samples)


def generate_forecasts(start_date, location, forecast_horizon):
    model_type = covid.models.SEIRD_incident.SEIRD
    
    util.gen_forecasts(data,
                       location,
                       model_type=model_type,
                       start=start_date,
                       end=None,
                       save=True,
                       show=False,
                       forecast_horizon=forecast_horizon)

# def get_final_r0(start_date, location):
#     zz = util.get_R_final(data,
#                    place,
#                    model_type=model_type,
#                    start=start,
#                    end=end,
#                    save=save,
#                    forecast_horizon=25)