import torch
import torch.nn as nn
import json
from pathlib import Path
from typing import Optional, Dict, Literal

ATOMIC_MASSES: Dict[int, float] = {
    1: 1.008,    # H
    6: 12.011,   # C
    7: 14.007,   # N
    8: 15.999,   # O
    9: 18.998,   # F
    14: 28.086,  # Si
    15: 30.974,  # P
    16: 32.065,  # S
    17: 35.453,  # Cl
    35: 79.904,  # Br
    53: 126.90,  # I
}

DEFAULT_MASS: float = 12.0


class MassWeightedCompactnessLoss(nn.Module):
    """
    Mass-weighted compactness loss based on radius of gyration.
    
    Computes: L = (1/M) * Σᵢ mᵢ||rᵢ - r_cm||²  (equals R_g²)
    
    Features:
        - Mass-weighted center of mass and covariance
        - Optional anisotropy penalty for spherical conformations
        - Optional timestep-dependent weighting for diffusion
        - Supports both batched (B, N, 3) and scattered formats
    """
    
    def __init__(
        self,
        atomic_masses: Optional[Dict[int, float]] = None,
        default_mass: float = DEFAULT_MASS,
        normalize_by_mass: bool = True,
        min_value: float = 1e-6,
        anisotropy_weight: float = 0.0,
        timestep_schedule: str = "none",
        t_max: int = 1000,
    ):
        """
        Args:
            atomic_masses: Mapping from atomic number to mass.
            default_mass: Mass for unknown elements.
            normalize_by_mass: Divide by total mass (standard R_g²).
            min_value: Minimum return value to prevent collapse.
            anisotropy_weight: Weight for anisotropy penalty (0 = disabled).
            timestep_schedule: "none", "linear", "cosine", or "sigmoid".
            t_max: Maximum timestep for schedule normalization.
        """
        super().__init__()
        
        self.atomic_masses = atomic_masses if atomic_masses is not None else ATOMIC_MASSES
        self.default_mass = default_mass
        self.normalize_by_mass = normalize_by_mass
        self.min_value = min_value
        self.anisotropy_weight = anisotropy_weight
        self.timestep_schedule = timestep_schedule
        self.t_max = t_max
        
        max_atomic_num = max(self.atomic_masses.keys()) + 1
        mass_lookup = torch.full((max_atomic_num,), default_mass)
        for z, m in self.atomic_masses.items():
            mass_lookup[z] = m
        self.register_buffer("mass_lookup", mass_lookup)
    
    def get_masses(self, atom_types: torch.Tensor) -> torch.Tensor:
        """Convert atomic numbers to masses."""
        clamped = atom_types.clamp(0, len(self.mass_lookup) - 1)
        masses = self.mass_lookup[clamped]
        
        out_of_range = (atom_types >= len(self.mass_lookup)) | (atom_types < 0)
        if out_of_range.any():
            masses = masses.clone()
            masses[out_of_range] = self.default_mass
        
        return masses
    
    def get_timestep_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Compute loss weight based on diffusion timestep."""
        t_normalized = t.float() / self.t_max
        
        if self.timestep_schedule == "none":
            return torch.ones_like(t_normalized)
        elif self.timestep_schedule == "linear":
            return 1.0 - t_normalized
        elif self.timestep_schedule == "cosine":
            return 0.5 * (1.0 + torch.cos(torch.pi * t_normalized))
        elif self.timestep_schedule == "sigmoid":
            return torch.sigmoid(10.0 * (0.5 - t_normalized))
        else:
            raise ValueError(f"Unknown timestep_schedule: {self.timestep_schedule}")
    
    def compute_anisotropy(self, cov: torch.Tensor) -> torch.Tensor:
        """Compute anisotropy = λ_max / Σλ from covariance matrix."""
        eigenvalues = torch.linalg.eigvalsh(cov)
        total = eigenvalues.sum(dim=-1) + 1e-8
        return eigenvalues[..., -1] / total
    
    def forward(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute mass-weighted compactness loss.
        
        Args:
            coords: (B, N, 3) batched or (N_total, 3) scattered
            atom_types: (B, N) or (N_total,) atomic numbers
            batch_idx: (N_total,) batch assignment for scattered format
            timestep: Diffusion timestep for schedule weighting
        """
        if batch_idx is None:
            return self._forward_batched(coords, atom_types, timestep)
        else:
            return self._forward_scattered(coords, atom_types, batch_idx, timestep)
    
    def _forward_batched(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Batched implementation (B, N, 3)."""
        B, N, _ = coords.shape
        
        masses = self.get_masses(atom_types.view(-1)).view(B, N)
        total_mass = masses.sum(dim=1, keepdim=True)
        
        com = (masses.unsqueeze(-1) * coords).sum(dim=1) / total_mass
        centered = coords - com.unsqueeze(1)
        
        weighted_centered = masses.unsqueeze(-1).sqrt() * centered
        cov = torch.bmm(weighted_centered.transpose(1, 2), weighted_centered)
        
        if self.normalize_by_mass:
            cov = cov / total_mass.unsqueeze(-1)
        else:
            cov = cov / N
        
        trace = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        trace = trace.clamp(min=self.min_value)
        
        if self.anisotropy_weight > 0:
            anisotropy = self.compute_anisotropy(cov)
            trace = trace + self.anisotropy_weight * (anisotropy - 1/3).clamp(min=0)
        
        if timestep is not None:
            trace = trace * self.get_timestep_weight(timestep)
        
        return trace.mean()
    
    def _forward_scattered(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        batch_idx: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scattered implementation (N_total, 3) with variable molecule sizes."""
        from torch_scatter import scatter_add
        
        num_graphs = batch_idx.max().item() + 1
        device = coords.device
        dtype = coords.dtype
        
        masses = self.get_masses(atom_types)
        total_mass = scatter_add(masses, batch_idx, dim=0, dim_size=num_graphs)
        
        weighted_coords = masses.unsqueeze(-1) * coords
        com_numerator = scatter_add(weighted_coords, batch_idx, dim=0, dim_size=num_graphs)
        com = com_numerator / total_mass.unsqueeze(-1).clamp(min=1e-8)
        
        centered = coords - com[batch_idx]
        sq_dist = (centered ** 2).sum(dim=-1)
        weighted_sq_dist = masses * sq_dist
        
        trace_numerator = scatter_add(weighted_sq_dist, batch_idx, dim=0, dim_size=num_graphs)
        
        if self.normalize_by_mass:
            trace = trace_numerator / total_mass.clamp(min=1e-8)
        else:
            ones = torch.ones(coords.shape[0], device=device, dtype=dtype)
            counts = scatter_add(ones, batch_idx, dim=0, dim_size=num_graphs)
            trace = trace_numerator / counts.clamp(min=1)
        
        trace = trace.clamp(min=self.min_value)
        
        if self.anisotropy_weight > 0:
            anisotropy_penalties = []
            for i in range(num_graphs):
                mask = (batch_idx == i)
                mol_centered = centered[mask]
                mol_masses = masses[mask]
                
                if mol_centered.shape[0] < 3:
                    anisotropy_penalties.append(torch.tensor(0.0, device=device, dtype=dtype))
                    continue
                
                weighted = mol_masses.unsqueeze(-1).sqrt() * mol_centered
                cov = weighted.T @ weighted
                if self.normalize_by_mass:
                    cov = cov / total_mass[i].clamp(min=1e-8)
                else:
                    cov = cov / mol_centered.shape[0]
                
                anisotropy = self.compute_anisotropy(cov.unsqueeze(0)).squeeze()
                anisotropy_penalties.append((anisotropy - 1/3).clamp(min=0))
            
            trace = trace + self.anisotropy_weight * torch.stack(anisotropy_penalties)
        
        if timestep is not None:
            trace = trace * self.get_timestep_weight(timestep)
        
        return trace.mean()
    
    def get_radius_of_gyration(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute mass-weighted R_g = sqrt(R_g²) for monitoring."""
        with torch.no_grad():
            if batch_idx is None:
                B, N, _ = coords.shape
                masses = self.get_masses(atom_types.view(-1)).view(B, N)
                total_mass = masses.sum(dim=1, keepdim=True)
                
                com = (masses.unsqueeze(-1) * coords).sum(dim=1) / total_mass
                centered = coords - com.unsqueeze(1)
                
                sq_dist = (centered ** 2).sum(dim=-1)
                rg_squared = (masses * sq_dist).sum(dim=1) / total_mass.squeeze(-1)
                return torch.sqrt(rg_squared.clamp(min=1e-8))
            else:
                from torch_scatter import scatter_add
                
                num_graphs = batch_idx.max().item() + 1
                masses = self.get_masses(atom_types)
                total_mass = scatter_add(masses, batch_idx, dim=0, dim_size=num_graphs)
                
                weighted_coords = masses.unsqueeze(-1) * coords
                com_numerator = scatter_add(weighted_coords, batch_idx, dim=0, dim_size=num_graphs)
                com = com_numerator / total_mass.unsqueeze(-1).clamp(min=1e-8)
                
                centered = coords - com[batch_idx]
                sq_dist = (centered ** 2).sum(dim=-1)
                rg_squared = scatter_add(masses * sq_dist, batch_idx, dim=0, dim_size=num_graphs)
                rg_squared = rg_squared / total_mass.clamp(min=1e-8)
                

                return torch.sqrt(rg_squared.clamp(min=1e-8))


class StatisticalProxyPotentialLoss:
    def __init__(self,
                 atom_mapping: dict = None,
                 r_min: float = 0.0,
                 r_max: float = 15.0,
                 r_n_bins: int = 1000,
                 spp_path: str | Path = "./potentials.json",
                 spp_key: Literal["spp", "spp_norm"] = "spp",
                 lj_min: float = -1.0,
                 lj_max: float = 1e3):
        self.atom_mapping = atom_mapping if atom_mapping is not None else {1: "H", 6: "C", 7: "N", 8: "O"}
        path = Path(spp_path)
        with path.open("r", encoding="utf-8") as f:
            spp_values = json.load(f)

        self.spp_dict = {}
        for pair in spp_values:
            self.spp_dict[pair] = torch.tensor(spp_values[pair][spp_key], dtype=torch.float32)

        self.r = torch.linspace(r_min, r_max, r_n_bins)
        self.lj_min = lj_min
        self.lj_max = lj_max


    def _interp1d_vectorized(self, x, xp, fp):
        idx = torch.searchsorted(xp, x).clamp(1, xp.numel() - 1)
        x0, x1 = xp[idx-1], xp[idx]
        y0, y1 = fp[idx-1], fp[idx]
        t = (x - x0) / (x1 - x0 + 1e-12)
        return y0 + t * (y1 - y0)


    def _spp_loss(self, all_coords, R_t, spp_dict):
        atoms = sorted(all_coords.keys())
        total_energy = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        pair_count = 0

        coord_tensors = {}
        for atom in atoms:
            c = all_coords[atom]
            if isinstance(c, torch.Tensor):
                coord_tensors[atom] = c
            else:
                coord_tensors[atom] = torch.tensor(c, dtype=torch.float32, requires_grad=True)

        for i in range(len(atoms)):
            for j in range(i+1, len(atoms)):
                a1, a2 = atoms[i], atoms[j]
                key = f"{a1}-{a2}"         # spp_dict "atom1-atom2", where atom1 < atom2
                A = coord_tensors[a1]      # (N1, 3)
                B = coord_tensors[a2]      # (N2, 3)

                # Vectorized distances
                diff = A[:, None, :] - B[None, :, :]
                dist = torch.sqrt((diff**2).sum(-1) + 1e-12)

                if key in spp_dict:
                    # Vectorized interpolation
                    U_curve = spp_dict[key]
                    U = self._interp1d_vectorized(dist, R_t, U_curve)
                    total_energy = total_energy + U.sum()
                else:
                    # Lennard-Jones potential as fallback
                    total_energy = total_energy + torch.clip((4 / (dist**12) - 4 / (dist**6)).sum(), min=self.lj_min, max=self.lj_max)
                pair_count += dist.numel()

        return total_energy / (pair_count + 1e-12)
    

    def forward(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mass-weighted compactness loss.
        
        Args:
            coords: (B, N, 3) batched or (N_total, 3) scattered
            atom_types: (B, N) or (N_total,) atomic numbers
        """
        total_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        b, n = atom_types.shape if len(atom_types.shape) == 2 else (1, atom_types.shape[0])
        for i in range(b):
            all_coords = {}
            for j in range(n):
                atom_int = atom_types[i][j].item() if len(atom_types.shape) == 2 else atom_types[i].item()
                if atom_int == 0:
                    break

                atom_type = self.atom_mapping.get(atom_int, str(atom_int))
                if atom_type not in all_coords:
                    all_coords[atom_type] = []
                all_coords[atom_type].append(coords[i][j] if len(coords.shape) == 3 else coords[i])

            for atom in all_coords:
                all_coords[atom] = torch.stack(all_coords[atom], dim=0)

            total_loss = total_loss + self._spp_loss(all_coords, self.r, self.spp_dict)
        return total_loss / b
