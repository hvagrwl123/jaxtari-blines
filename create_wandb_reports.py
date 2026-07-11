import wandb_workspaces.reports.v2 as wr

ENTITY = "raef-tu-darmstadt"

def game_section(title, our_project, our_query, our_metrics, cleanrl_query=None, table_md="", caption=""):
    runsets = [wr.Runset(entity=ENTITY, project=our_project, name="ours", query=our_query)]
    if cleanrl_query:
        runsets.append(wr.Runset(entity="openrlbenchmark", project="cleanrl",
                                 name="CleanRL reference", query=cleanrl_query))
    panels = [wr.LinePlot(title=t, y=[m], smoothing_factor=0.8)
              for t, m in our_metrics]
    blocks = [wr.H2(text=title),
              wr.PanelGrid(runsets=runsets, panels=panels)]
    if table_md:
        blocks.append(wr.MarkdownBlock(text=table_md))
    if caption:
        blocks.append(wr.MarkdownBlock(text=f"*{caption}*"))
    return blocks

# ---------------- PQN report ----------------
pqn_params = """
| Parameter | CleanRL pqn_atari_envpool | original | tuned |
|---|---|---|---|
| parallel envs | 8 | 8 | 64 |
| rollout length | 128 | 128 | 128 |
| batch per update | 1,024 | 1,024 | 8,192 |
| minibatches / epochs | 4 / 4 | 4 / 4 | 32 / 4 |
| learning rate (RAdam) | 2.5e-4, annealed | same | same |
| Q(lambda) | 0.65 | 0.65 | 0.65 |
| exploration fraction / final eps | 0.10 / 0.01 | same | same |
"""
pqn_metrics = [("training episodic return", "charts/episodic_return"),
               ("running average (last 20 episodes)", "charts/avg_episodic_return")]

blocks = [wr.MarkdownBlock(text=pqn_params),
          wr.MarkdownBlock(text="*Metrics logged once per chunk (32 iters): curves start at 33k (original) / 262k (tuned) steps.*")]
for game, clq, cap in [
    ("Pong", "Pong-v5__pqn_atari_envpool", "CleanRL @10M: 20.49 +/- 0.11. All four variants match."),
    ("Frostbite", None, "No CleanRL PQN reference."),
    ("Qbert", None, "No CleanRL PQN reference."),
    ("BeamRider", "BeamRider-v5__pqn_atari_envpool", "CleanRL raw curve plateaus near 2k; table value 5,753 +/- 2,395 uses different aggregation. JAXAtari scoring likely differs from ALE."),
    ("MsPacman", "MsPacman-v5__pqn_atari_envpool", "CleanRL @10M: 2,298.83 +/- 128.24; our pixel original: 2,297.5."),
]:
    blocks += game_section(game, "jaxatari-pqn-cleanrl", game.lower(), pqn_metrics, clq)

report = wr.Report(entity=ENTITY, project="jaxatari-pqn-cleanrl",
                   title="PQN on JAXAtari: results across five games (CleanRL-exact config)",
                   description="OC/pixel x original/tuned, 10M steps, seed 0. Original config identical to CleanRL pqn_atari_envpool defaults.",
                   blocks=blocks)
report.save()
print("PQN report:", report.url)

# ---------------- C51 report ----------------
c51_params = """
| Parameter | CleanRL c51_atari_jax | original | tuned |
|---|---|---|---|
| learning rate | 2.5e-4 | 2.5e-4 | 1e-4 |
| batch size | 32 | 32 | 64 |
| target update freq | 10,000 | 10,000 | 5,000 |
| exploration fraction / final eps | 0.10 / 0.01 | same | 0.25 / 0.05 |
| train frequency | 4 | 4 | 4 |
| learning starts | 80,000 | 80,000 (pixel) / 10,000 (OC) | same |
| buffer size | 1,000,000 | 100k (pixel) / 200k (OC) | 100k / 400k |
| V range / atoms | [-10,10] / 51 | same | same |
| parallel envs | 1 | 32 (replay ratio preserved) | 32 |
"""
c51_metrics = [("evaluation return (10 ep, eps=0.05)", "charts/episodic_return"),
               ("training return (reward-clipped)", "charts/avg_episodic_return")]

blocks = [wr.MarkdownBlock(text=c51_params),
          wr.MarkdownBlock(text="*Our charts/episodic_return is the unclipped eval; CleanRL's metric of the same name is their training return - scale-compatible. Our training returns (second panel) are reward-clipped.*")]
for game, clq, cap in [
    ("Pong", "PongNoFrameskip-v4__c51_atari_jax", "CleanRL @10M: 19.88 +/- 0.31."),
    ("Frostbite", None, "No CleanRL C51 reference."),
    ("Qbert", None, "No CleanRL C51 reference."),
    ("Seaquest", None, "No CleanRL C51 reference."),
    ("BeamRider", "BeamRiderNoFrameskip-v4__c51_atari_jax", "CleanRL @10M: 9,504.91 +/- 709.69."),
]:
    blocks += game_section(game, "jaxatari-c51-final", game.lower(), c51_metrics, clq)

report = wr.Report(entity=ENTITY, project="jaxatari-c51-final",
                   title="C51 on JAXAtari: results across five games",
                   description="OC/pixel x original/tuned, 10M steps, seed 0. Original config matches CleanRL c51_atari_jax except documented buffer/env deviations.",
                   blocks=blocks)
report.save()
print("C51 report:", report.url)
