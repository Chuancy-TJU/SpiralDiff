from . import gaussian_diffusion as gd
# import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps, SpacedDiffusionSpiralDiff, SpacedDiffusionSpiralDiff_y0

def create_gaussian_diffusion(
    *,
    normalize_input,
    schedule_name,
    sf=4,
    min_noise_level=0.01,
    steps=1000,
    kappa=1,
    etas_end=0.99,
    schedule_kwargs=None,
    weighted_mse=False,
    predict_type='xstart',
    timestep_respacing=None,
    scale_factor=None,
    latent_flag=True,
    guidence_weight=0,
    diffusion_kwargs=None,
):
    sqrt_etas = gd.get_named_eta_schedule(
            schedule_name,
            num_diffusion_timesteps=steps,
            min_noise_level=min_noise_level,
            etas_end=etas_end,
            kappa=kappa,
            kwargs=schedule_kwargs,
            )
    if timestep_respacing is None:
        timestep_respacing = steps
    else:
        assert isinstance(timestep_respacing, int)
    if predict_type == 'xstart':
        model_mean_type = gd.ModelMeanType.START_X
    elif predict_type == 'epsilon':
        model_mean_type = gd.ModelMeanType.EPSILON
    elif predict_type == 'epsilon_scale':
        model_mean_type = gd.ModelMeanType.EPSILON_SCALE
    elif predict_type == 'residual':
        model_mean_type = gd.ModelMeanType.RESIDUAL
    else:
        raise ValueError(f'Unknown Predicted type: {predict_type}')
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        sqrt_etas=sqrt_etas,
        kappa=kappa,
        model_mean_type=model_mean_type,
        loss_type=gd.LossType.WEIGHTED_MSE if weighted_mse else gd.LossType.MSE,
        scale_factor=scale_factor,
        normalize_input=normalize_input,
        sf=sf,
        latent_flag=latent_flag,
        guidence_weight=guidence_weight,
        diffusion_kwargs=diffusion_kwargs,
    )

def create_gaussian_diffusion_SpiralDiff(
    *,
    normalize_input,
    schedule_name,
    sf=1,
    min_noise_level=0.01,
    steps=1000,
    kappa=1,
    etas_end=0.99,
    schedule_kwargs=None,
    weighted_mse=False,
    predict_type='xstart',
    timestep_respacing=None,
    scale_factor=None,
    latent_flag=True,
    guidence_weight=0,
    diffusion_kwargs=None,
    zt_min=0.1,
):
    sqrt_etas = gd.get_named_eta_schedule(
            schedule_name,
            num_diffusion_timesteps=steps,
            min_noise_level=min_noise_level,
            etas_end=etas_end,
            kappa=kappa,
            kwargs=schedule_kwargs,
            )
    if timestep_respacing is None:
        timestep_respacing = steps
    else:
        assert isinstance(timestep_respacing, int)
    if predict_type == 'xstart':
        model_mean_type = gd.ModelMeanType.START_X
    elif predict_type == 'epsilon':
        model_mean_type = gd.ModelMeanType.EPSILON
    elif predict_type == 'epsilon_scale':
        model_mean_type = gd.ModelMeanType.EPSILON_SCALE
    elif predict_type == 'residual':
        model_mean_type = gd.ModelMeanType.RESIDUAL
    else:
        raise ValueError(f'Unknown Predicted type: {predict_type}')
    return SpacedDiffusionSpiralDiff(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        sqrt_etas=sqrt_etas,
        kappa=kappa,
        model_mean_type=model_mean_type,
        loss_type=gd.LossType.WEIGHTED_MSE if weighted_mse else gd.LossType.MSE,
        scale_factor=scale_factor,
        normalize_input=normalize_input,
        sf=sf,
        latent_flag=latent_flag,
        guidence_weight=guidence_weight,
        zt_min=zt_min,
        diffusion_kwargs=diffusion_kwargs,
    )

def create_gaussian_diffusion_y0(
    *,
    normalize_input,
    schedule_name,
    sf=1,
    min_noise_level=0.01,
    steps=1000,
    kappa=1,
    etas_end=0.99,
    schedule_kwargs=None,
    weighted_mse=False,
    predict_type='xstart',
    timestep_respacing=None,
    scale_factor=None,
    latent_flag=True,
    guidence_weight=0,
    diffusion_kwargs=None,
    zt_min=0.1,
):
    sqrt_etas = gd.get_named_eta_schedule(
            schedule_name,
            num_diffusion_timesteps=steps,
            min_noise_level=min_noise_level,
            etas_end=etas_end,
            kappa=kappa,
            kwargs=schedule_kwargs,
            )
    if timestep_respacing is None:
        timestep_respacing = steps
    else:
        assert isinstance(timestep_respacing, int)
    if predict_type == 'xstart':
        model_mean_type = gd.ModelMeanType.START_X
    elif predict_type == 'epsilon':
        model_mean_type = gd.ModelMeanType.EPSILON
    elif predict_type == 'epsilon_scale':
        model_mean_type = gd.ModelMeanType.EPSILON_SCALE
    elif predict_type == 'residual':
        model_mean_type = gd.ModelMeanType.RESIDUAL
    else:
        raise ValueError(f'Unknown Predicted type: {predict_type}')
    return SpacedDiffusionSpiralDiff_y0(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        sqrt_etas=sqrt_etas,
        kappa=kappa,
        model_mean_type=model_mean_type,
        loss_type=gd.LossType.WEIGHTED_MSE if weighted_mse else gd.LossType.MSE,
        scale_factor=scale_factor,
        normalize_input=normalize_input,
        sf=sf,
        latent_flag=latent_flag,
        guidence_weight=guidence_weight,
        zt_min=zt_min,
        diffusion_kwargs=diffusion_kwargs,
    )
