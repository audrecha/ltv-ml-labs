"""Lab 3: BenchMARL task for the portfolio environment."""
import copy
from typing import Callable, Dict, List, Optional

from benchmarl.environments.common import Task, TaskClass
from torchrl.data import Composite
from torchrl.envs import EnvBase

from env import PortfolioEnv, load_panel

PANELS = {}


def panel(name):
    if name not in PANELS:
        PANELS[name] = load_panel(name)
    return PANELS[name]


class PortfolioClass(TaskClass):
    def get_env_fun(self, num_envs: int, continuous_actions: bool, seed: Optional[int], device) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        universe = config.pop('universe', 'sectors')
        return lambda: PortfolioEnv(panel(universe), num_envs=num_envs, continuous=continuous_actions,
                                    seed=seed or 0, device=device, **config)

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return not self.config.get('single', False)

    def has_render(self, env: EnvBase) -> bool:
        return False

    def max_steps(self, env: EnvBase) -> int:
        return env.episode_len

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def observation_spec(self, env: EnvBase) -> Composite:
        spec = env.full_observation_spec_unbatched.clone()
        del spec['state']
        return spec

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return Composite({'state': env.full_observation_spec_unbatched['state'].clone()})

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec_unbatched

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    @staticmethod
    def env_name() -> str:
        return 'portfolio'


class PortfolioTask(Task):
    SECTORS = None

    @staticmethod
    def associated_class():
        return PortfolioClass
