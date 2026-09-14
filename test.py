import asyncio

from poke_env.player import RandomPlayer
from poke_env import AccountConfiguration, ShowdownServerConfiguration
from config import LOGIN,LOGIN_2, PASSWORD, CUSTOM_SERVER
from stable_baselines3 import PPO
from agents.policy_player_simple import PolicyPlayer, MaskedActorCriticPolicy, BATTLE_FORMAT


async def randomPlay():
    player = RandomPlayer(
    account_configuration=AccountConfiguration(LOGIN, PASSWORD),
    server_configuration=CUSTOM_SERVER,
    battle_format="gen9fusionmonsrandombattle")
    while True:
        await player.accept_challenges(None,2)

async def aiPlay():
    ppo = PPO.load("models/self_play_snapshot_1800k")
    player = PolicyPlayer(avatar="schoolkid-gen4dp",account_configuration=AccountConfiguration(LOGIN_2, PASSWORD),policy=ppo.policy,server_configuration=CUSTOM_SERVER, battle_format=BATTLE_FORMAT)
    while True:
        await player.ladder(2)
        

if __name__ == "__main__":
    #asyncio.run(randomPlay())
    asyncio.run(aiPlay())
