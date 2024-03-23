import json
from collections import defaultdict

from aprl.envs.gym_compete import NUM_ZOO_POLICIES, env_name_to_canonical

from ap.envs.gym_compete.env_gym_compete import ENV_LIST

victim_zoo = defaultdict(dict)
for env_name in ENV_LIST:
    for i in range(NUM_ZOO_POLICIES[env_name_to_canonical(env_name)]):
        victim_zoo[env_name].update({f"zoo{i+1}": dict(policy_type="zoo", tag=i + 1)})
    victim_zoo[env_name].update({"zero": dict(policy_type="cityu", method="zero")})
    victim_zoo[env_name].update({"rand": dict(policy_type="cityu", method="rand")})

with open("victim_zoo.json", "w") as f:
    json.dump(victim_zoo, f, indent=4)
