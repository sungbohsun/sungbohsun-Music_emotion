import wandb
api = wandb.Api()
runs = api.runs('Emotion')
runs[3].group = runs[3].name[:-2]
runs[3].update()

#'WANDB_API_KEY'] = '*******************************'