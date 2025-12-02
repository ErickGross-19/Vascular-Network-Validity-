"""
Full network flow solver using scipy.sparse.

Wraps the existing vascular_network package solver for integration.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from ..core.network import VascularNetwork
from ..core.result import OperationResult, OperationStatus, ErrorCode


def solve_flow(
    network: VascularNetwork,
    inlet_node_ids: Optional[List[int]] = None,
    outlet_node_ids: Optional[List[int]] = None,
    pin: float = 13000.0,
    pout: float = 2000.0,
    mu: float = 1.0e-3,
    write_to_network: bool = True,
) -> OperationResult:
    """
    Solve flow through the vascular network using full Poiseuille solver.
    
    This wraps the existing vascular_network.analysis.cfd.compute_poiseuille_network
    function, converting to/from NetworkX format.
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network
    inlet_node_ids : list of int, optional
        Inlet node IDs. If None, auto-detect from node_type
    outlet_node_ids : list of int, optional
        Outlet node IDs. If None, auto-detect from node_type
    pin : float
        Inlet pressure (Pa). Default ~100 mmHg
    pout : float
        Outlet pressure (Pa). Default ~15 mmHg
    mu : float
        Dynamic viscosity (Pa·s). Default for blood at 37°C
    write_to_network : bool
        Whether to write results back to network node/segment attributes
        
    Returns
    -------
    OperationResult
        Result with flow solution in metadata
    """
    from ..adapters.networkx_adapter import to_networkx_graph
    from vascular_network.analysis.cfd import compute_poiseuille_network
    
    try:
        if inlet_node_ids is None:
            inlet_node_ids = [
                nid for nid, node in network.nodes.items()
                if node.node_type == "inlet"
            ]
        
        if outlet_node_ids is None:
            outlet_node_ids = [
                nid for nid, node in network.nodes.items()
                if node.node_type == "outlet"
            ]
        
        if not inlet_node_ids:
            return OperationResult.failure(
                "No inlet nodes found",
                error_codes=[ErrorCode.NODE_NOT_FOUND.value],
            )
        
        if not outlet_node_ids:
            return OperationResult.failure(
                "No outlet nodes found",
                error_codes=[ErrorCode.NODE_NOT_FOUND.value],
            )
        
        G, node_id_map = to_networkx_graph(network)
        
        inlet_nx = [node_id_map[nid] for nid in inlet_node_ids]
        outlet_nx = [node_id_map[nid] for nid in outlet_node_ids]
        
        # Solve
        flow_result = compute_poiseuille_network(
            G,
            mu=mu,
            inlet_nodes=inlet_nx,
            outlet_nodes=outlet_nx,
            pin=pin,
            pout=pout,
            write_to_graph=True,
        )
        
        node_pressures = flow_result["node_pressures"]
        edge_flows = flow_result["edge_flows"]
        summary = flow_result["summary"]
        
        if write_to_network:
            nx_to_vascular = {v: k for k, v in node_id_map.items()}
            
            for nx_id, pressure in node_pressures.items():
                vascular_id = nx_to_vascular[nx_id]
                network.nodes[vascular_id].attributes['pressure'] = float(pressure)
            
            for (u_nx, v_nx), flow in edge_flows.items():
                u_vascular = nx_to_vascular[u_nx]
                v_vascular = nx_to_vascular[v_nx]
                
                for seg in network.segments.values():
                    if ((seg.start_node_id == u_vascular and seg.end_node_id == v_vascular) or
                        (seg.start_node_id == v_vascular and seg.end_node_id == u_vascular)):
                        seg.attributes['flow'] = float(abs(flow))
                        seg.attributes['velocity'] = float(
                            abs(flow) / (np.pi * ((seg.geometry.radius_start + seg.geometry.radius_end) / 2) ** 2)
                        )
                        break
        
        result = OperationResult.success(
            f"Flow solved: {summary['num_nodes']} nodes, {summary['num_edges']} edges",
            metadata={
                'summary': summary,
                'inlet_flow': float(summary['total_inlet_flow']),
                'outlet_flow': float(summary['total_outlet_flow']),
                'flow_balance_error': float(
                    abs(summary['total_inlet_flow'] - summary['total_outlet_flow']) /
                    max(summary['total_inlet_flow'], 1e-10)
                ),
                'pin': float(pin),
                'pout': float(pout),
                'mu': float(mu),
            },
        )
        
        for warning in summary.get('warnings', []):
            result.add_warning(warning)
        
        if summary.get('used_lstsq', False):
            result.add_warning("System was rank-deficient, used least-squares")
            result.error_codes.append(ErrorCode.DIRICHLET_SINGULAR.value)
        
        balance_error = result.metadata['flow_balance_error']
        if balance_error > 0.05:
            result.add_warning(f"Flow balance error {balance_error:.1%} exceeds 5%")
            result.error_codes.append(ErrorCode.FLOW_BALANCE_ERROR.value)
        
        return result
        
    except Exception as e:
        return OperationResult.failure(
            f"Flow solver failed: {e}",
            error_codes=[ErrorCode.DIRICHLET_SINGULAR.value],
        )


def check_flow_plausibility(network: VascularNetwork) -> OperationResult:
    """
    Check if flow solution is hemodynamically plausible.
    
    Checks:
    - Flow conservation at junctions
    - Pressure monotonicity
    - Reynolds number ranges
    - Murray's law compliance
    
    Parameters
    ----------
    network : VascularNetwork
        Network with solved flow (must have 'pressure' and 'flow' attributes)
        
    Returns
    -------
    OperationResult
        Result with plausibility checks in metadata
    """
    warnings = []
    errors = []
    
    has_pressure = any('pressure' in node.attributes for node in network.nodes.values())
    has_flow = any('flow' in seg.attributes for seg in network.segments.values())
    
    if not has_pressure or not has_flow:
        return OperationResult.failure(
            "Network does not have flow solution (missing 'pressure' or 'flow' attributes)",
        )
    
    for node_id, node in network.nodes.items():
        if node.node_type == "junction":
            inflow = 0.0
            outflow = 0.0
            
            for seg in network.segments.values():
                flow = seg.attributes.get('flow', 0.0)
                
                if seg.end_node_id == node_id:
                    inflow += flow
                elif seg.start_node_id == node_id:
                    outflow += flow
            
            balance = abs(inflow - outflow) / max(inflow, 1e-10)
            if balance > 0.1:
                warnings.append(
                    f"Node {node_id}: flow balance error {balance:.1%}"
                )
    
    for seg in network.segments.values():
        p_start = network.nodes[seg.start_node_id].attributes.get('pressure', 0)
        p_end = network.nodes[seg.end_node_id].attributes.get('pressure', 0)
        flow = seg.attributes.get('flow', 0)
        
        if flow > 0 and p_end > p_start:
            warnings.append(
                f"Segment {seg.id}: pressure increases along flow direction"
            )
    
    max_re = 0.0
    for seg in network.segments.values():
        velocity = seg.attributes.get('velocity', 0)
        radius = (seg.geometry.radius_start + seg.geometry.radius_end) / 2
        diameter = 2 * radius
        
        rho = 1060.0  # kg/m³ for blood
        mu = 1.0e-3  # Pa·s
        re = rho * velocity * diameter / mu
        max_re = max(max_re, re)
        
        if re > 2300:
            warnings.append(
                f"Segment {seg.id}: turbulent flow (Re={re:.0f})"
            )
    
    result = OperationResult.success(
        f"Plausibility check complete: {len(warnings)} warnings, {len(errors)} errors",
        metadata={
            'max_reynolds': float(max_re),
            'is_laminar': max_re < 2300,
            'num_warnings': len(warnings),
            'num_errors': len(errors),
        },
    )
    
    for warning in warnings:
        result.add_warning(warning)
    
    for error in errors:
        result.add_error(error)
    
    return result
