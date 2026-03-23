import ast

from dataclasses import field, make_dataclass

from pyhanami.utils import data_general
from pyhanami.config import config_params

PARAMETERS = data_general.load_yaml_file(config_params.SCI_EVAL_PARAMS_PATH)

TYPE_MAP = {
    "int": int,
    "float": float,
    "str": str,
    "tuple": tuple,
    "list": list,
    "dict": dict,
    "bool": bool,
}


def parse_value(value, expected_type_str):
    """
    Parse a value ensuring it matches the expected type.

    Parameters
    ----------
    value : any
        Value to parse.
    expected_type_str : str
        Expected Python type of the value as a string.

    Returns
    -------
    parsed_value : any
        Parsed value, cast to the expected type.
    """
    aux_value = value
    expected_type = TYPE_MAP[expected_type_str]


    # Handle already correct type
    if isinstance(value, expected_type):
        pass

    # Handle structured types
    elif expected_type_str in ["tuple", "list", "dict"]:
        try:
            aux_value = ast.literal_eval(value)
        except Exception as e:
            raise TypeError(f"Value '{value}' cannot be parsed as type '{expected_type_str}'. Error: {e}")

    # Handle float expressions
    elif expected_type_str == "float":
        try:
            aux_value = float(value)
        except ValueError:
            aux_value = eval(value, {"__builtins__": {}}, {})   # Use eval in a restricted environment for safety

    # Handle boolean values
    elif expected_type_str == "bool":
        if isinstance(value, str):
            if value.lower() in ['true', '1']:
                aux_value = True
            elif value.lower() in ['false', '0']:
                aux_value = False
            else:
                raise ValueError(f"String value '{value}' cannot be parsed as a boolean.")

    
    # Cast to the expected type
    try: 
        parsed_value = expected_type(aux_value)
    except Exception as e:
        raise TypeError(f"Value '{value}' cannot be parsed as type '{expected_type_str}'. Error: {e}")
    
    return parsed_value


def make_config_class(phenomenon):
    """
    Create configuration dataclass with default parameters necessary to compute 
    scores for the given phenomenon.

    Parameters
    ----------
    phenomenon : str
        Name of the phenomenon to evaluate.

    Returns
    -------
    ConfigClass : dataclass
        Dataclass with one field for each parameter necessary to evaluate
        the given phenomenon, with default values taken from the PARAMETERS 
        dictionary.
    """

    # Validate input
    if phenomenon not in PARAMETERS:
        raise ValueError(f"Phenomenon '{phenomenon}' not found in '{config_params.SCI_EVAL_PARAMS_PATH}'. "
                         f"Available phenomena: {list(PARAMETERS.keys())}")
    

    # Load parameters
    phenomenon_params = PARAMETERS[phenomenon]
    dataclass_fields = []

    for param_name, param_info in phenomenon_params.items():
        # Validate parameter metadata
        if 'value' not in param_info or 'type' not in param_info:
            raise ValueError(f"Parameter '{param_name} in phenomenon '{phenomenon}' must have 'value' and "
                             f"'type' fields in the configuration file '{config_params.SCI_EVAL_PARAMS_PATH}'.")

        # Load parameter value and type
        param_value = param_info['value']
        param_type_str = param_info['type']
        
        if param_type_str not in TYPE_MAP:
            raise ValueError(f"Unsupported type '{param_type_str}' for parameter '{param_name}' in phenomenon "
                             f"'{phenomenon}' in the configuration file '{config_params.SCI_EVAL_PARAMS_PATH}'. "
                             f"Supported types: {list(TYPE_MAP.keys())}")
        param_type = TYPE_MAP[param_type_str]

        # Parse value as the expected type
        default_value = parse_value(param_value, param_type_str)
        
        # Add parameter to dataclass fields
        dataclass_fields.append((param_name, param_type, field(default=default_value)))


    # Validate types of user-provided parameters
    def __post_init__(self):
        for param, spec in phenomenon_params.items():
            if not hasattr(self, param):
                raise ValueError(f"Missing parameter '{param}' for phenomenon '{phenomenon}'")
        
            current_value = getattr(self, param)
            expected_type_str = spec['type']

            try:
                parsed_value = parse_value(current_value, expected_type_str)
                setattr(self, param, parsed_value)
            except Exception as e:
                raise TypeError(f"Parameter '{param}' in configuration for phenomenon '{phenomenon}' must be of type "
                                f"'{expected_type_str}', but got value '{current_value}' of type "
                                f"'{type(current_value).__name__}'. Error: {e}")

            # if not isinstance(current_value, expected_type):
            #     # Allow integers for float parameters
            #     if expected_type == float and isinstance(current_value, int):
            #         continue  

    # Create dataclass with the loaded and validated parameters
    class_name = f"{phenomenon.upper()}Config"
    ConfigClass = make_dataclass(
        cls_name=class_name,
        fields=dataclass_fields,
        namespace={'__post_init__': __post_init__}
    )

    return ConfigClass


# Create configuration dataclasses for each phenomenon
ISOConfig = make_config_class('iso')
MJOConfig = make_config_class('mjo')
TCConfig = make_config_class('tc')