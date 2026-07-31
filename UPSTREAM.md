# Upstream provenance and experiment map

## Source snapshot

- Upstream: <https://github.com/wushulin/ActaPinT>
- Branch inspected: `main`
- Pinned commit: `68431b7`
- Companion article: <https://doi.org/10.1017/S0962492924000072>
- Upstream language: MATLAB
- Python conversion location: `actapint/`

The upstream snapshot contains 23 MATLAB files. They are retained at repository
root so numerical choices can be checked against the original scripts.

## Implemented mapping

| Python entry point | Upstream MATLAB source | Paper coverage |
|---|---|---|
| `solution_heat_ade` | `Solution_Heat_ADE.m` | model solutions in Section 2 |
| `solution_burgers` | `Solution_Bergers.m` | model solutions in Section 2 |
| `solution_wave` | `Solution_Wave.m` | model solutions in Section 2 |
| `parareal_heat_ade` | `Parareal_Heat_ADE.m` | Figure 4.5, linear case |
| `parareal_burgers` | `Parareal_Burgers.m` | Figure 4.5, nonlinear case |
| `mgrit_heat_ade` | `MGRiT_Heat_ADE.m` | Figures 4.9–4.10 |
| `iterative_paradiag_ade` | `Iterative_ParaDiag_ADE.m` | Figure 3.15(c,d) |
| `stmg_heat_ade` | `STMG_Heat_ADE.m` | Figures 4.19–4.20 |
| `paper_validation` wave case | `Iterative_ParaDiag_Wave.m`, adjusted to the paper parameters | Figure 3.15(e,f) |

The unified paper-validation entry point combines matched parameter sweeps and
fair-cost comparisons:

```bash
actapint paper_validation --output-dir results/paper
```

## Indexed but not yet converted

The remaining scripts cover SWR, IDC/PIDC, ParaExp, direct ParaDiag, diagonal
coarse-grid correction, and space-time domain decomposition. They are kept in
the repository as the extension backlog:

- `SWR_for_ADE.m`, `UTP_DD_Method.m`, `RedBlackSubdomains.m`
- `IDC_PIDC_ADE.m`, `ParaExp_Burgers.m`
- `Direct_ParaDiag_ADE.m`, `Direct_ParaDiag_Burgers.m`,
  `Direct_ParaDiag_Wave.m`
- `PCGC_Parareal_Heat_ADE.m`, `PCGC_Parareal_Burgers.m`,
  `DiagG_Parareal_Heat_Wave.m`
- `Iterative_ParaDiag_Wave.m`, `Iterative_ParaDiag_RoundError.m`
- `MGRiT_Burgers.m`, `STMG_Bergers.m`

## Known source correction

`MGRiT_Heat_ADE.m` contains `nu=0.002max;`, which is not valid MATLAB. The
Python port uses `nu=0.002`, matching the corresponding branch, title, and
parameter sweep in the paper.
