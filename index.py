import asyncio

from poke_env.player import RandomPlayer
from poke_env import AccountConfiguration, ShowdownServerConfiguration
from config import LOGIN,LOGIN_2, PASSWORD, CUSTOM_SERVER
from stable_baselines3 import PPO
from agents.policy_player import PolicyPlayer, MaskedActorCriticPolicy, BATTLE_FORMAT
import io
import zipfile
import torch
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.save_util import load_from_zip_file

async def randomPlay():
    player = RandomPlayer(
    account_configuration=AccountConfiguration(LOGIN, PASSWORD),
    server_configuration=CUSTOM_SERVER,
    battle_format="gen9fusionmonsrandombattle")
    while True:
        await player.accept_challenges(None,2)

async def aiPlay():
    print("1. Извлечение пространств Gym напрямую из файла модели...")
    try:
        # load_from_zip_file возвращает (data, pytorch_variables, verbose_json)
        # Этот метод прочитает метаданные без инициализации тяжелых C-модулей
        serialized_data, _, _ = load_from_zip_file("models/pretrained_10000.zip")
        
        # Автоматически забираем пространства, на которых обучалась модель
        obs_space = serialized_data["observation_space"]
        act_space = serialized_data["action_space"]
        print(f"📊 Успешно извлечено! Obs: {obs_space}, Act: {act_space}")
    except Exception as e:
        print(f"❌ Не удалось прочитать метаданные из zip: {e}")
        return

    print("2. Создание экземпляра политики MaskedActorCriticPolicy...")
    try:
        # Инициализируем пустую нейросеть с точными размерами из обучения
        policy_instance = MaskedActorCriticPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=lambda _: 0.0
        )
        policy_instance.to("cpu")
        policy_instance.eval()
    except Exception as e:
        print(f"❌ Ошибка инициализации класса политики: {e}")
        return

    print("3. Загрузка бинарных весов (state_dict)...")
    try:
        with zipfile.ZipFile("models/pretrained_10000.zip", "r") as archive:
            policy_bytes = archive.read("policy.pth")
            buffer = io.BytesIO(policy_bytes)
            state_dict = torch.load(buffer, map_location="cpu")
        
        # Накатываем веса тензоров прямо на созданную политику
        policy_instance.load_state_dict(state_dict)
        print("🤖 Веса модели успешно применены!")
    except Exception as e:
        print(f"❌ Ошибка при импорте тензоров весов: {e}")
        return

    print("4. Инициализация PolicyPlayer...")
    # Передаем готовую и наполненную весами политику в вашего игрока
    player = PolicyPlayer(
        avatar="schoolkid-gen4dp",
        account_configuration=AccountConfiguration(LOGIN_2, PASSWORD),
        policy=policy_instance,  # Теперь здесь лежит полноценный рабочий объект нейросети
        server_configuration=CUSTOM_SERVER, 
        battle_format=BATTLE_FORMAT,
        start_timer_on_battle_start=True,
    )

    print("🚀 Бот полностью готов! Ожидание входящих вызовов...")
    await player.ps_client.wait_for_login()
    await player.ps_client.send_message("/join lobby")
    await player.ps_client.send_message("Привет. Я бот, способный сыграть в Fusionmons Random Battle. Просто вызови меня на бой в этом формате и я с тобой сыграю!", room="lobby")
    while True:
        try:
            # Передаем None во второй аргумент. 
            # Бот будет держать WebSocket открытым и принимать ВСЕ вызовы один за другим.
            await player.accept_challenges(None, 1)
            await asyncio.sleep(5)
        except Exception as e:
            # Сюда теперь прилетит ConnectionClosedError, если сеть упадет
            print(f"💥 Соединение разорвано ({e}). Переподключение через 5 секунд...")
            player._battles.clear()
            await asyncio.sleep(5)
            
async def aiPlay2():
    print("1. Извлечение пространств Gym напрямую из файла модели...")
    try:
        # load_from_zip_file возвращает (data, pytorch_variables, verbose_json)
        # Этот метод прочитает метаданные без инициализации тяжелых C-модулей
        serialized_data, _, _ = load_from_zip_file("models/pretrained_10000.zip")
        
        # Автоматически забираем пространства, на которых обучалась модель
        obs_space = serialized_data["observation_space"]
        act_space = serialized_data["action_space"]
        print(f"📊 Успешно извлечено! Obs: {obs_space}, Act: {act_space}")
    except Exception as e:
        print(f"❌ Не удалось прочитать метаданные из zip: {e}")
        return

    print("2. Создание экземпляра политики MaskedActorCriticPolicy...")
    try:
        # Инициализируем пустую нейросеть с точными размерами из обучения
        policy_instance = MaskedActorCriticPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=lambda _: 0.0
        )
        policy_instance.to("cpu")
        policy_instance.eval()
    except Exception as e:
        print(f"❌ Ошибка инициализации класса политики: {e}")
        return

    print("3. Загрузка бинарных весов (state_dict)...")
    try:
        with zipfile.ZipFile("models/pretrained_10000.zip", "r") as archive:
            policy_bytes = archive.read("policy.pth")
            buffer = io.BytesIO(policy_bytes)
            state_dict = torch.load(buffer, map_location="cpu")
        
        # Накатываем веса тензоров прямо на созданную политику
        policy_instance.load_state_dict(state_dict)
        print("🤖 Веса модели успешно применены!")
    except Exception as e:
        print(f"❌ Ошибка при импорте тензоров весов: {e}")
        return

    print("4. Инициализация PolicyPlayer...")
    # Передаем готовую и наполненную весами политику в вашего игрока
    player = PolicyPlayer(
        avatar="clown",
        account_configuration=AccountConfiguration(LOGIN, PASSWORD),
        policy=policy_instance,  # Теперь здесь лежит полноценный рабочий объект нейросети
        server_configuration=CUSTOM_SERVER, 
        battle_format="gen9randombattle",
        start_timer_on_battle_start=True,
    )

    print("🚀 Бот полностью готов! Ожидание входящих вызовов...")
    await player.ps_client.wait_for_login()
    await player.ps_client.send_message("/join lobby")
    await player.ps_client.send_message("Привет! Я бот, способный сыграть в Random Battle. Просто вызови меня на бой в этом формате и я с тобой сыграю!", room="lobby")
    while True:
        try:
            # Передаем None во второй аргумент. 
            # Бот будет держать WebSocket открытым и принимать ВСЕ вызовы один за другим.
            await player.accept_challenges(None, 1)
            await asyncio.sleep(5)
        except Exception as e:
            # Сюда теперь прилетит ConnectionClosedError, если сеть упадет
            print(f"💥 Соединение разорвано ({e}). Переподключение через 5 секунд...")
            player._battles.clear()
            await asyncio.sleep(5)

async def main():
    # Запускаем обе функции параллельно
    await asyncio.gather(
        aiPlay(),
        aiPlay2()
    )

if __name__ == "__main__":
    asyncio.run(main())