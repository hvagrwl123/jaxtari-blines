import os
import wandb

api = wandb.Api()
os.makedirs("report_data", exist_ok=True)

for run in api.runs("raef-tu-darmstadt/jaxatari-c51-final"):
    h = run.history(samples=2000)
    h.to_csv(f"report_data/{run.name}.csv", index=False)
    print("exported", run.name, run.state)

for env in ["PongNoFrameskip-v4", "BeamRiderNoFrameskip-v4"]:
    runs = api.runs("openrlbenchmark/cleanrl",
                    {"config.env_id": env, "config.exp_name": "c51_atari_jax"})
    for r in runs:
        h = r.history(keys=["global_step", "charts/episodic_return"], samples=2000)
        h.to_csv(f"report_data/cleanrl_{env}_seed{r.config.get('seed')}.csv", index=False)
        print("exported cleanrl", env, "seed", r.config.get("seed"))
