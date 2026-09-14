from poke_env.environment import SingleAgentWrapper, SinglesEnv,PokeEnv

import inspect
print([m for m in dir(PokeEnv) if "message" in m.lower() or "handle" in m.lower()])