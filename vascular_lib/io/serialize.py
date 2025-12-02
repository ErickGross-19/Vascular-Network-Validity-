"""
JSON serialization for vascular networks.
"""

import json
from pathlib import Path
from typing import Union
from ..core.network import VascularNetwork


def save_json(
    network: VascularNetwork,
    filepath: Union[str, Path],
    indent: int = 2,
) -> None:
    """
    Save vascular network to JSON file.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to save
    filepath : str or Path
        Output file path
    indent : int
        JSON indentation level
    
    Example
    -------
    >>> from vascular_lib import save_json
    >>> save_json(network, "my_network.json")
    """
    filepath = Path(filepath)
    
    data = network.to_dict()
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def load_json(filepath: Union[str, Path]) -> VascularNetwork:
    """
    Load vascular network from JSON file.
    
    Parameters
    ----------
    filepath : str or Path
        Input file path
    
    Returns
    -------
    network : VascularNetwork
        Loaded network
    
    Example
    -------
    >>> from vascular_lib import load_json
    >>> network = load_json("my_network.json")
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    schema_version = data.get("schema_version", "1.0")
    if schema_version != "1.0":
        raise ValueError(f"Unsupported schema version: {schema_version}")
    
    network = VascularNetwork.from_dict(data)
    
    return network
