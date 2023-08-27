"""
The methods in config.py follow, with slight variations, the work of:

Braker, J., Heinemann, L., Schreieder, T.: Aramis at touché 2022: Argument detection in pictures using machine learning.
Working Notes Papers of the CLEF (2022)

Carnot, M.L., Heinemann, L., Braker, J., Schreieder, T., Kiesel, J., Fröbe,
M., Potthast, M., Stein, B.: On Stance Detection in Image Retrieval for Argumentation. In: Proc. of SIGIR. ACM (2023)

Schreieder, T., Braker, J.: Touché 2022 Best of Labs: Neural Image Retrieval for Argumentation. CLEF (2023)

The original source code can be found at: https://github.com/webis-de/SIGIR-23
"""

import json
import logging
from pathlib import Path

log = logging.getLogger('Config')


class Config:

    data_dir: Path = Path('data/')
    output_dir: Path = Path('out/')
    working_dir: Path = Path('working/')
    data_image_format: bool = False
    on_win: bool = False

    _save_path = Path('config.json')

    _cfg = None

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(working_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def get(cls) -> 'Config':
        cfg = cls()
        if Config._cfg is not None:
            return Config._cfg
        if Config._save_path.exists():
            try:
                with open(Config._save_path, 'r') as f:
                    cfg_json = json.load(f)
                cfg.data_dir = Path(cfg_json.get('data_dir', cfg.data_dir))
                cfg.output_dir = Path(cfg_json.get('output_dir', cfg.output_dir))
                cfg.working_dir = Path(cfg_json.get('working_dir', cfg.working_dir))
                cfg.data_image_format = bool(cfg_json.get('image_format', cfg.data_image_format))
                cfg.on_win = bool(cfg_json.get('on_win', cfg.on_win))
            except json.JSONDecodeError:
                pass
        cfg.save()
        log.debug('Config loaded')

        Config._cfg = cfg
        return cfg

    def save(self) -> None:
        if not Config._save_path.exists():
            log.debug('Config saved.')
            with open(Config._save_path, 'w+') as f:
                json.dump(self.to_dict(), f)

    def to_dict(self) -> dict:
        return {
            'data_dir': str(self.data_dir),
            'output_dir': str(self.output_dir),
            'working_dir': str(self.working_dir),
            'image_format': self.data_image_format,
            'on_win': self.on_win
        }
