#!/bin/bash
CL="++alg.NUM_STEPS=128 ++alg.UPDATE_EPOCHS=4 ++alg.ANNEAL_LR=True ++alg.END_E=0.01"
for GAME in pong frostbite qbert beamrider mspacman; do
  uv run python main.py +alg=pqn_oc_original ++alg.ENV_ID=$GAME $CL ++alg.RUN_NAME=oc_original_$GAME ++PROJECT=jaxatari-pqn-cleanrl
  uv run python main.py +alg=pqn_oc_tuned ++alg.ENV_ID=$GAME $CL ++alg.RUN_NAME=oc_tuned_$GAME ++PROJECT=jaxatari-pqn-cleanrl
  uv run python main.py +alg=pqn_rgb_original ++alg.ENV_ID=$GAME $CL ++alg.RUN_NAME=pixel_original_$GAME ++PROJECT=jaxatari-pqn-cleanrl
  uv run python main.py +alg=pqn_rgb_tuned ++alg.ENV_ID=$GAME $CL ++alg.RUN_NAME=pixel_tuned_$GAME ++PROJECT=jaxatari-pqn-cleanrl
done
