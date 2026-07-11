import os
import wandb

api = wandb.Api()
os.makedirs("pqn_cl_data", exist_ok=True)

for run in api.runs("raef-tu-darmstadt/jaxatari-pqn-cleanrl"):
    if run.state != "finished":
        print("skipping", run.name, run.state)
        continue
    h = run.history(samples=2000)
    h.to_csv(f"pqn_cl_data/{run.name}.csv", index=False)
    print("exported", run.name)

for env in ["Pong-v5", "BeamRider-v5", "MsPacman-v5"]:
    runs = api.runs("openrlbenchmark/cleanrl",
                    {"config.env_id": env, "config.exp_name": "pqn_atari_envpool"})
    for r in runs:
        h = r.history(keys=["global_step", "charts/episodic_return"], samples=2000)
        h.to_csv(f"pqn_cl_data/cleanrl_{env}_seed{r.config.get('seed')}.csv", index=False)
        print("exported cleanrl", env, "seed", r.config.get("seed"))
