from ui.base import create_ui
from utils.args_parser import parse_args
from utils.config import load_config
from utils.logger import setup_logger

# Streamlit wont execute if its in a main block.
args = parse_args()
cfg = load_config(args.config)

# Create the actual UI
create_ui(cfg)
