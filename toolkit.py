#!/usr/bin/env python3
"""
toolkit.py
==========
Modular toolkit for Single-Cell RNA-seq (scRNA-seq) analysis and modeling.
Developed from the computational methods of Master's Thesis (TFM):
"Computational approaches to study the maintenance of neuronal identity
and plasticity during aging".

Author: Pablo Manuel Lopez Ruiz
GitHub: https://github.com/pabloLopezRuiz
"""

from typing import Dict, List, Optional, Tuple, Union, Set, Any
import textwrap
import numpy as np
import pandas as pd
import scipy.stats as sci
import scipy.sparse as sp
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import seaborn as sns
from adjustText import adjust_text


# =============================================================================
#   SCIENTIFIC STYLES, PALETTES AND CONFIGURATION
# =============================================================================

# Selected color palette
COLORES = pd.Series({
    "sig": "#D62728",      # Red for significant target group genes
    "nosig": "#1F77B4",    # Blue for non-significant
    "sub": "#ff9900",       # Orange for non-sig. target group genes
    "bg": "#7F7F7F",       # Dark gray background
    "bg2": "#d9d9d9",      # Light gray background
})

# Gene category labels used across the script
SUB_NAME: str = "Subgroup of TFs"
TF_NAME: str = "Transcription factors"
REST_NAME: str = "Rest of genes"
# Maximum number of characters of a plot label before skiping to the next line
MAX_LABEL_CHARS: int = 15


def _parse_time_values(series: pd.Series) -> np.ndarray:
    """
    Extract numeric values from a time series (e.g. ints, floats, 'd1', 'd11').
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input 'series' must be a pandas Series.")
    if len(series) == 0:
        raise ValueError("Input time series must not be empty.")

    try:
        vals: np.ndarray = series.astype(float).values
    except (ValueError, TypeError):
        cleaned: pd.Series = (
            series.astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0]
        )
        vals = cleaned.astype(float).values

    if np.all(np.isnan(vals)):
        raise ValueError(
            "Failed to extract valid numeric time values from series."
        )
    return vals


def apply_tfm_style() -> None:
    """
    Configure matplotlib global parameters.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "figure.titlesize": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })
apply_tfm_style()


# =============================================================================
#   QUALITY CONTROL AND PREPROCESSING (QC)
# =============================================================================

def qc_and_preprocess(
    adata: sc.AnnData,
    mt_col: Optional[str] = None,
    ribo_col: Optional[str] = None,
    max_mt_pct: float = 10.0,
    max_ribo_pct: float = 10.0,
    min_genes_per_cell: int = 200,
    min_cells_per_gene: int = 2,
    target_sum: float = 1e4,
    log_transform: bool = True,
) -> Dict[str, Any]:
    """
    Compute QC metrics and plot figures, filter dataset,
    and apply normalization.

    Parameters
    ----------
    adata : sc.AnnData
        Input AnnData object with count data.
    mt_col : str, optional
        Column in adata.var indicating mitochondrial genes.
    ribo_col : str, optional
        Column in adata.var indicating ribosomal genes.
    max_mt_pct : float, default 10.0
        Maximum allowed percentage of mitochondrial counts per cell.
    max_ribo_pct : float, default 10.0
        Maximum allowed percentage of ribosomal counts per cell.
    min_genes_per_cell : int, default 200
        Minimum number of genes required to retain a cell.
    min_cells_per_gene : int, default 2
        Minimum number of cells expressing a gene.
    target_sum : float, default 10000.0
        Target total count per cell for library normalization.
    log_transform : bool, default True
        If True, applies sc.pp.log1p (log(x + 1)).

    Returns
    -------
    Dict[str, Any]
        Dictionary with QC metrics and figures ('fig_violin',
        'fig_depth_pre', 'fig_depth_post').
    """
    # Input validation
    if not isinstance(adata, sc.AnnData):
        raise TypeError("Input 'adata' must be a scanpy AnnData object.")
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("Input 'adata' must not be empty.")
    if max_mt_pct < 0.0 or max_ribo_pct < 0.0:
        raise ValueError("Percentage thresholds must be non-negative.")
    if min_genes_per_cell < 0 or min_cells_per_gene < 0:
        raise ValueError("Filter count thresholds must be non-negative.")
    if target_sum <= 0.0:
        raise ValueError("target_sum must be a positive number.")

    # Detect available mitochondrial and ribosomal annotation columns
    qc_vars: List[str] = []
    if mt_col and mt_col in adata.var.columns:
        qc_vars.append(mt_col)
    if ribo_col and ribo_col in adata.var.columns:
        qc_vars.append(ribo_col)

    # Compute cell-level quality metrics (counts, genes, fractions)
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=qc_vars, inplace=True, percent_top=None
    )

    # Plot distribution of mt and ribo genes if annotation columns exist
    fig_violin: Optional[plt.Figure] = None
    ax_violin: Optional[plt.Axes] = None
    if len(qc_vars) > 0:
        fig_violin, ax_violin = plt.subplots(figsize=(6, 4))
        violin_cols: List[str] = [
            f"pct_counts_{var}"
            for var in qc_vars
            if f"pct_counts_{var}" in adata.obs.columns
        ]

        sc.pl.violin(adata, violin_cols, jitter=0.4, ax=ax_violin, show=False)
        # Draw threshold reference lines
        if mt_col and mt_col in adata.var.columns:
            ax_violin.axhline(
                y=max_mt_pct, color="darkgrey", linestyle="--", linewidth=1.5
            )
        if ribo_col and ribo_col in adata.var.columns:
            ax_violin.axhline(
                y=max_ribo_pct, color="darkred", linestyle="--", linewidth=1.5
            )

        has_mt: bool = bool(mt_col and mt_col in adata.var.columns)
        has_ribo: bool = bool(ribo_col and ribo_col in adata.var.columns)
        if has_mt and has_ribo:
            ax_violin.set_title(
                "Mitochondrial and Ribosomal Genes Distribution"
            )
        elif has_mt:
            ax_violin.set_title("Mitochondrial Genes Distribution")
        else:
            ax_violin.set_title("Ribosomal Genes Distribution")
        ax_violin.set_ylabel("Percentage")
        labels: List[str] = violin_cols
        ax_violin.set_xticks(range(len(labels)))
        ax_violin.set_xticklabels(labels)
        fig_violin.tight_layout()
    # End plot

    # Plot pre-normalization sequencing depth
    fig_depth_pre, ax_depth_pre = plt.subplots(figsize=(6, 4))
    sc.pl.scatter(
        adata, x="total_counts", y="n_genes_by_counts",
        ax=ax_depth_pre, size=50, show=False
    )
    ax_depth_pre.set_title("Pre-normalization Sequencing Depth")
    ax_depth_pre.set_xlabel("Sequencing Depth (total_counts)")
    ax_depth_pre.tick_params(axis="x", labelrotation=45, labelsize=10)
    ax_depth_pre.set_ylabel("Detected Genes (n_genes_by_counts)")
    fig_depth_pre.tight_layout()
    # End plot

    n_cells_before: int = int(adata.n_obs)
    n_genes_before: int = int(adata.n_vars)

    # Filter out cells failing quality control thresholds
    mt_mask: pd.Series = pd.Series(False, index=adata.obs_names)
    ribo_mask: pd.Series = pd.Series(False, index=adata.obs_names)
    if mt_col and f"pct_counts_{mt_col}" in adata.obs:
        mt_mask = adata.obs[f"pct_counts_{mt_col}"] >= max_mt_pct
    if ribo_col and f"pct_counts_{ribo_col}" in adata.obs:
        ribo_mask = adata.obs[f"pct_counts_{ribo_col}"] >= max_ribo_pct
    keep_cells: pd.Series = (
        (~(mt_mask | ribo_mask)) &
        (adata.obs["n_genes_by_counts"] > min_genes_per_cell)
    )
    adata._inplace_subset_obs(keep_cells.values)

    # Filter unexpressed or sparsely expressed genes in-place
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene, inplace=True)
    n_cells_after: int = int(adata.n_obs)
    n_genes_after: int = int(adata.n_vars)

    # Scale library sizes to target depth and recalculate metrics
    sc.pp.normalize_total(adata, target_sum=target_sum, inplace=True)
    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None)

    # Plot post-normalization sequencing depth
    fig_depth_post, ax_depth_post = plt.subplots(figsize=(6, 4))
    sc.pl.scatter(
        adata, x="total_counts", y="n_genes_by_counts",
        ax=ax_depth_post, size=50, show=False
    )
    ax_depth_post.set_title("Post-normalization Sequencing Depth")
    ax_depth_post.set_xlabel("Sequencing Depth (total_counts)")
    ax_depth_post.tick_params(axis="x", labelrotation=45, labelsize=10)
    ax_depth_post.set_ylabel("Detected Genes (n_genes_by_counts)")
    fig_depth_post.tight_layout()
    # End plot

    # Apply log1p transformation to stabilize variance
    if log_transform:
        sc.pp.log1p(adata)

    # Close plots to prevent unwanted display
    if fig_violin is not None:
        plt.close(fig_violin)
    plt.close(fig_depth_pre)
    plt.close(fig_depth_post)
    
    results: Dict[str, Any] = {
        "metrics": {
            "n_cells_before": n_cells_before,
            "n_cells_after": n_cells_after,
            "n_genes_before": n_genes_before,
            "n_genes_after": n_genes_after,
            "cells_removed_mt": int(mt_mask.sum()),
            "cells_removed_ribo": int(ribo_mask.sum()),
        },
        "fig_violin": fig_violin,
        "fig_depth_pre": fig_depth_pre,
        "fig_depth_post": fig_depth_post,
    }
    return results


# =============================================================================
#   EXPLORATORY ANALYSIS: SAMPLE DISTRIBUTIONS
# =============================================================================

def plot_sample_distributions(
    adata: sc.AnnData,
    disc_key: Optional[str] = "timepoint",
    cont_key: Optional[str] = "cell_type",
    p_inf: float = 5.0,
    p_sup: float = 95.0,
    time_key: Optional[str] = None,
    celltype_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Plot sample distributions across two .obs columns: a discrete column
    (barplot) and a continuous frequency density column (KDE plot).

    Parameters
    ----------
    adata : sc.AnnData
        Input AnnData object.
    disc_key : str, optional, default 'timepoint'
        Column in .obs representing the discrete variable.
    cont_key : str, optional, default 'cell_type'
        Column in .obs representing the continuous/density variable.
    p_inf : float, default 5.0
        Lower percentile to highlight in density plot.
    p_sup : float, default 95.0
        Upper percentile to highlight in density plot.
    time_key : str, optional
        Deprecated alias for disc_key.
    celltype_key : str, optional
        Deprecated alias for cont_key.

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'counts_disc', 'fig_disc', 'counts_cont',
        'stats_cont', 'fig_cont').
    """
    # Input validation
    if not isinstance(adata, sc.AnnData):
        raise TypeError("Input 'adata' must be a scanpy AnnData object.")
    if not (0.0 <= p_inf < p_sup <= 100.0):
        raise ValueError(
            "Percentile bounds must satisfy 0 <= p_inf < p_sup <= 100."
        )
    if disc_key is not None and disc_key not in adata.obs.columns:
        raise KeyError(
            f"disc_key '{disc_key}' not found in adata.obs."
        )
    if cont_key is not None and cont_key not in adata.obs.columns:
        raise KeyError(
            f"cont_key '{cont_key}' not found in adata.obs."
        )

    # Plot sample distribution across discrete column
    df_disc: Optional[pd.Series] = None
    fig_disc: Optional[plt.Figure] = None
    ax_disc: Optional[plt.Axes] = None
    if disc_key and disc_key in adata.obs.columns:
        df_disc = adata.obs[disc_key].value_counts().sort_index()
        fig_disc, ax_disc = plt.subplots(figsize=(6, 4))
        wrapped_labels: List[str] = [
            textwrap.fill(str(val), width=MAX_LABEL_CHARS)
            for val in df_disc.index
        ]
        sns.barplot(
            x=wrapped_labels, y=df_disc.values,
            color=COLORES["nosig"], width=0.6, ax=ax_disc
        )
        ax_disc.set_xlabel(str(disc_key))
        ax_disc.set_ylabel("Number of cells")
        ax_disc.set_title(f"Sample Distribution Across {disc_key}")
        fig_disc.tight_layout()
    # End plot

    # Plot distribution across continuous/frequency column
    counts_cont: Optional[pd.Series] = None
    stats_cont: Optional[Dict[str, float]] = None
    fig_cont: Optional[plt.Figure] = None
    ax_cont: Optional[plt.Axes] = None
    if cont_key and cont_key in adata.obs.columns:
        counts_cont = adata.obs[cont_key].value_counts()
        p_lower: float = float(np.percentile(counts_cont, p_inf))
        p_upper: float = float(np.percentile(counts_cont, p_sup))
        mean_count: float = float(counts_cont.mean())

        fig_cont, ax_cont = plt.subplots(figsize=(6, 4))
        sns.kdeplot(counts_cont, fill=True, color=COLORES["sig"], ax=ax_cont)
        # Use invisible line handle to display the mean in the legend
        ax_cont.plot([], [], " ", label=f"Mean: {mean_count:.1f}")
        ax_cont.axvline(
            p_lower, color=COLORES["bg"], linestyle="--",
            label=f"$P_{{{int(p_inf)}}}$: {p_lower:.1f}"
        )
        ax_cont.axvline(
            p_upper, color=COLORES["bg"], linestyle="--",
            label=f"$P_{{{int(p_sup)}}}$: {p_upper:.1f}"
        )
        ax_cont.legend(handlelength=0)
        ax_cont.set_xlabel(f"Cells per {cont_key}")
        ax_cont.set_ylabel("Density")
        ax_cont.set_title(f"Density Distribution of {cont_key}")
        fig_cont.tight_layout()

        stats_cont = {
            "mean": mean_count, "p_inf": p_lower, "p_sup": p_upper
        }
    # Close plots to prevent unwanted display
    if fig_disc is not None:
        plt.close(fig_disc)
    if fig_cont is not None:
        plt.close(fig_cont)

    results: Dict[str, Any] = {
        "counts_disc": df_disc,
        "fig_disc": fig_disc,
        "counts_cont": counts_cont,
        "stats_cont": stats_cont,
        "fig_cont": fig_cont
    }
    return results


# =============================================================================
#   DIMENSIONALITY REDUCTIONS (PCA / UMAP)
# =============================================================================

def compute_dimred(
    adata: sc.AnnData,
    n_pcs: int = 30,
    n_neighbors: int = 15,
    n_components_umap: int = 2,
    color_keys: Optional[Union[str, List[str]]] = None,
    rewrite: bool = True,
) -> Dict[str, Any]:
    """
    Compute PCA, explained variance ratio, neighborhood graph, and UMAP
    in-place.

    Parameters
    ----------
    adata : sc.AnnData
        Normalized AnnData object.
    n_pcs : int, default 30
        Number of principal components to compute.
    n_neighbors : int, default 15
        Number of neighbors for cell connectivity graph.
    n_components_umap : int, default 2
        Dimensions of the resulting UMAP embedding.
    color_keys : str or list of str, optional
        Columns in .obs used to color PCA and UMAP projections.
    rewrite : bool, default True
        If True, overwrites existing embeddings.

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'variance_ratio', 'fig_variance', 'fig_pca',
        and 'fig_umap'.
    """
    # Input validation
    if not isinstance(adata, sc.AnnData):
        raise TypeError("Input 'adata' must be a scanpy AnnData object.")
    if n_pcs <= 0 or n_neighbors <= 0 or n_components_umap <= 0:
        raise ValueError(
            "n_pcs, n_neighbors, and n_components_umap must be positive."
        )
    if isinstance(color_keys, str):
        color_keys = [color_keys]

    # Compute PCA if needed
    if rewrite or "X_pca" not in adata.obsm:
        sc.tl.pca(adata, n_comps=n_pcs)

    # Extract explained variance ratios
    var_exp: np.ndarray = adata.uns["pca"]["variance_ratio"][:n_pcs]
    pcs: np.ndarray = np.arange(len(var_exp)) + 1

    # Plot explained variance
    fig_var, ax_var = plt.subplots(figsize=(6, 4))
    ax_var.bar(pcs, var_exp, color=COLORES["nosig"])
    ax_var.set_xlabel("PC")
    ax_var.set_ylabel("Explained variance")
    ax_var.set_title("Explained Variance by Principal Component")
    fig_var.tight_layout()
    # End plot

    # Compute UMAP
    if rewrite or "X_umap" not in adata.obsm:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        sc.tl.umap(adata, n_components=n_components_umap)

    valid_keys: Optional[List[str]] = None
    if color_keys and len(color_keys) > 0:
        valid_keys = [k for k in color_keys if k in adata.obs.columns]

    # Plot PCA
    fig_pca = sc.pl.pca(
        adata,
        legend_loc="best",
        color=valid_keys,
        show=False,
        return_fig=True
    )
    # End plot

    # Plot UMAP
    fig_umap = sc.pl.umap(
        adata, 
        color=valid_keys, 
        legend_loc="best",
        show=False, 
        return_fig=True
    )
    # Close plots to prevent unwanted display
    if fig_var is not None:
        plt.close(fig_var)
    if fig_pca is not None:
        plt.close(fig_pca)
    if fig_umap is not None:
        plt.close(fig_umap)

    results: Dict[str, Any] = {
        "variance_ratio": var_exp,
        "fig_variance": fig_var,
        "fig_pca": fig_pca,
        "fig_umap": fig_umap,
    }
    return results


# =============================================================================
#   DIFFERENTIAL EXPRESSION ANALYSIS (OLD vs. YOUNG)
# =============================================================================

def run_stratified_dea(
    adata: sc.AnnData,
    grp_key: str = "cell_type",
    condition_key: str = "age",
    target_group: str = "old",
    reference_group: str = "young",
    sel_method: str = "wilcoxon",
    min_cells_per_condition: int = 5,
    pval_thresh: float = 0.05,
    sub_dict: Optional[Dict[str, Set[str]]] = None,
    tf_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform stratified DEA between two conditions (target~reference) within
    each defined cell population.

    Parameters
    ----------
    adata : sc.AnnData
        Normalized AnnData object (can be pre-filtered to target genes).
    grp_key : str, default 'cell_type'
        Column in .obs defining distinct cell groups.
    condition_key : str, default 'age'
        Column in .obs with conditions to contrast.
    target_group : str, default 'old'
        Target group of interest.
    reference_group : str, default 'young'
        Reference baseline group.
    sel_method : str, default 'wilcoxon'
        Statistical test method used by Scanpy.
    min_cells_per_condition : int, default 5
        Minimum cells per condition required to run the contrast.
    pval_thresh : float, default 0.05
        Significance threshold for adjusted p-values.
    sub_dict : Dict[str, Set[str]], optional
        Dictionary mapping cell groups to their established subset of genes.
    tf_col : Optional[str], optional
            Optional column in adata.var to annotate significant TFs in summary.


    Returns
    -------
    Dict[str, Any]
        Dictionary with 'dea_by_group' and 'summary_table'.
    """
    # Input validation
    if not isinstance(adata, sc.AnnData):
        raise TypeError("Input 'adata' must be a scanpy AnnData object.")
    if grp_key not in adata.obs.columns:
        raise KeyError(f"grp_key column '{grp_key}' not found in adata.obs.")
    if condition_key not in adata.obs.columns:
        raise KeyError(
            f"condition_key column '{condition_key}' not found in adata.obs."
        )
    cond_vals = set(adata.obs[condition_key].dropna().astype(str).unique())
    if str(target_group) not in cond_vals:
        raise ValueError(
            f"target_group '{target_group}' not found in "
            f"adata.obs['{condition_key}']."
        )
    if str(reference_group) not in cond_vals:
        raise ValueError(
            f"reference_group '{reference_group}' not found in "
            f"adata.obs['{condition_key}']."
        )
    if min_cells_per_condition < 1:
        raise ValueError("min_cells_per_condition must be at least 1.")
    if not (0.0 < pval_thresh <= 1.0):
        raise ValueError("pval_thresh must be between 0 and 1.")

    res_dict: Dict[str, pd.DataFrame] = {}
    unique_groups: np.ndarray = adata.obs[grp_key].dropna().unique()

    # Load transcription factor identifiers if column provided
    tf_set: Set[str] = set()
    if tf_col and tf_col in adata.var.columns:
        tf_set = set(adata.var_names[adata.var[tf_col]])

    # Ensure the condition column is formatted as a categorical series
    adata.obs[condition_key] = (
        adata.obs[condition_key]
        .astype(str)
        .astype("category")
    )

    # Define the target and reference group
    target_group_str: str = str(target_group)
    ref_group_str: str = str(reference_group)

    # Perform stratified differential expression analysis for each cell group
    for grp in unique_groups:
        # Subset of the current cell group
        sub_adata: sc.AnnData = adata[adata.obs[grp_key] == grp].copy()
        cond_counts: pd.Series = sub_adata.obs[condition_key].value_counts()

        # Ensure sufficient cells in each category
        if (
            target_group_str not in cond_counts or
            ref_group_str not in cond_counts or
            cond_counts[target_group_str] < min_cells_per_condition or
            cond_counts[ref_group_str] < min_cells_per_condition
        ):
            continue

        # Differential expression analysis
        sc.tl.rank_genes_groups(
            sub_adata,
            groupby=condition_key,
            groups=[target_group_str],
            reference=ref_group_str,
            method=sel_method,
            use_raw=False,
        )
        df_dea: pd.DataFrame = sc.get.rank_genes_groups_df(
            sub_adata, group=target_group_str
        )
        df_dea.set_index("names", inplace=True)
        res_dict[str(grp)] = df_dea

    # Summary table for DE results
    summary_rows: List[Dict[str, Any]] = []
    for grp, df in res_dict.items():
        sig_msk: pd.Series = df["pvals_adj"] < pval_thresh
        sig_genes: int = int(sig_msk.sum())
        row: Dict[str, Any] = {
            "Cell Group": grp,
            "Total Genes": len(df),
            "Sig. Genes": sig_genes,
        }

        # Annotate TF counts if tf_set available
        if len(tf_set) > 0:
            tf_msk: np.ndarray = df.index.isin(tf_set)
            row["Sig. TF"] = int((tf_msk & sig_msk).sum())

        # Check if any of the celltype subset of genes is DE
        if sub_dict:
            sub_genes: Set[str] = sub_dict.get(grp, set())
            sub_present: List[str] = list(sub_genes.intersection(df.index))
            sig_sub: int = int(
                (df.loc[sub_present, "pvals_adj"] < pval_thresh).sum()
            )
            row["Total target"] = len(sub_present)
            row["Sig. Target"] = sig_sub

        summary_rows.append(row)

    df_summary: pd.DataFrame = pd.DataFrame(summary_rows)

    results: Dict[str, Any] = {
        "dea_by_group": res_dict,
        "summary_table": df_summary
    }
    return results


# =============================================================================
#   VOLCANO PLOT
# =============================================================================

def plot_volcano(
    dea_df: pd.DataFrame,
    target_genes: Optional[Union[Set[str], List[str]]] = None,
    pval_col: str = "pvals_adj",
    lfc_col: str = "logfoldchanges",
    gene_label_col: Optional[str] = "common",
    pval_thresh: float = 0.05,
    lfc_thresh: float = 1.0,
    lfc_highlight_thresh: float = 4.0,
    title: str = "Volcano Plot",
    xlims: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = (8, 5),
) -> Dict[str, Any]:
    """
    Plot Volcano Plot with 6 scientific categories, cutoff lines, and labels.

    Parameters
    ----------
    dea_df : pd.DataFrame
        Differential expression results table.
    target_genes : set or list of str, optional
        Target gene identifiers.
    pval_col : str, default 'pvals_adj'
        Adjusted p-value column name.
    lfc_col : str, default 'logfoldchanges'
        Log2 fold change column name.
    gene_label_col : str, optional
        Column containing labels to display.
    pval_thresh : float, default 0.05
        Significance cutoff threshold.
    lfc_thresh : float, default 1.0
        Log2 fold change cutoff threshold.
    lfc_highlight_thresh : float, default 4.0
        LFC threshold for labeling additional non-target genes.
    title : str, default 'Volcano Plot'
        Plot title.
    xlims : tuple, optional
        X-axis limits.
    figsize : tuple, default (8, 5)
        Figure dimensions.

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'fig', 'ax', and 'classified_df'.
    """
    # Input validation
    if not isinstance(dea_df, pd.DataFrame):
        raise TypeError("Input 'dea_df' must be a pandas DataFrame.")
    if dea_df.empty:
        raise ValueError("Input 'dea_df' cannot be empty.")
    if pval_col not in dea_df.columns:
        raise KeyError(f"pval_col '{pval_col}' not found in dea_df.")
    if lfc_col not in dea_df.columns:
        raise KeyError(f"lfc_col '{lfc_col}' not found in dea_df.")
    if pval_thresh <= 0.0:
        raise ValueError("pval_thresh must be positive.")
    if lfc_thresh < 0.0 or lfc_highlight_thresh < 0.0:
        raise ValueError("Fold change thresholds must be non-negative.")

    # Coerce target_genes to set
    target_genes_set: Set[str] = (
        set(target_genes) if target_genes is not None else set()
    )

    # Volcano plot (volcano)
    fig, ax = plt.subplots(figsize=figsize)

    # Filter missing values and transform p-values to -log10 scale
    df: pd.DataFrame = dea_df.dropna(subset=[pval_col, lfc_col]).copy()
    clipped_pvals: pd.Series = df[pval_col].clip(lower=1e-300)
    df["-logpadj"] = -np.log10(clipped_pvals)

    # Establish condition masks
    p_mask: pd.Series = df[pval_col] < pval_thresh
    l_mask: pd.Series = np.abs(df[lfc_col]) > lfc_thresh
    is_target: np.ndarray = df.index.isin(target_genes_set)

    # Stratify genes into categories
    df["label"] = "Not significant"
    df.loc[p_mask, "label"] = "Significant"
    df.loc[p_mask & l_mask, "label"] = f"Significant + |logFC| > {lfc_thresh}"
    df.loc[is_target, "label"] = "Target not significant"
    df.loc[is_target & p_mask, "label"] = "Target significant"
    df.loc[
        is_target & p_mask & l_mask, "label"
    ] = f"Target sig. + |logFC| > {lfc_thresh}"

    attr: List[Tuple[str, str, float, int, str]] = [
        ("Not significant", COLORES["bg"], 0.25, 12, "o"),
        ("Significant", COLORES["nosig"], 0.45, 20, "o"),
        (
            f"Significant + |logFC| > {lfc_thresh}",
            COLORES["nosig"], 0.55, 30, "^"
        ),
        ("Target not significant", COLORES["sub"], 0.85, 45, "o"),
        ("Target significant", COLORES["sig"], 0.95, 65, "o"),
        (
            f"Target sig. + |logFC| > {lfc_thresh}",
            COLORES["sig"], 0.95, 75, "^"
        ),
    ]

    # Volcano plot, category by category
    for lbl, color, alpha, size, marker in attr:
        sub: pd.DataFrame = df[df["label"] == lbl]
        if len(sub) > 0:
            ax.scatter(
                sub[lfc_col], sub["-logpadj"],
                c=color, alpha=alpha, s=size, marker=marker, label=lbl
            )

    # Draw lines for significance and fold-change thresholds
    ax.axhline(
        -np.log10(pval_thresh), color=COLORES["sig"],
        ls="dashed", lw=1.2, alpha=0.7
    )
    ax.axvline(lfc_thresh, color=COLORES["bg"], ls="dashed", lw=1.2, alpha=0.7)
    ax.axvline(
        -lfc_thresh, color=COLORES["bg"], ls="dashed", lw=1.2, alpha=0.7
    )
    texts: List[plt.Text] = []
    # Plot texts
    is_sub_sig: pd.Series = df["label"].isin([
        "Target significant",
        f"Target sig. + |logFC| > {lfc_thresh}"
    ])
    is_high_lfc: pd.Series = (
        (df[pval_col] < pval_thresh) &
        (df[lfc_col].abs() > lfc_highlight_thresh)
    )
    candidates: pd.DataFrame = df[is_sub_sig | is_high_lfc]

    for idx_val, row in candidates.iterrows():
        lbl_text: str = (
            str(row[gene_label_col])
            if (gene_label_col and gene_label_col in row)
            else str(idx_val)
        )
        texts.append(ax.text(
            row[lfc_col], row["-logpadj"], lbl_text,
            fontsize=9, fontweight="bold"
        ))

    if xlims:
        ax.set_xlim(xlims)
    ax.set_xlabel("Change (log2FoldChange)")
    ax.set_ylabel("-log10(adj p-value)")
    ax.set_title(title)
    ax.legend(frameon=True, loc="best", fontsize=8)
    fig.tight_layout()

    # Close plots to prevent unwanted display
    plt.close(fig)

    results: Dict[str, Any] = {
        "fig": fig,
        "ax": ax,
        "classified_df": df,
    }
    return results


# =============================================================================
#   TEMPORAL TRENDS AND LINEAR MODELING (SLOPES)
# =============================================================================

def compute_temporal_slopes(
    adata: sc.AnnData,
    time_key: str = "timepoint",
    min_expressing_cells: int = 10,
    sub_col: Optional[str] = "is_sub",
    tf_col: Optional[str] = "is_tf"
) -> Dict[str, Any]:
    """
    Compute temporal expression slope for each gene across time in-place.

    Parameters
    ----------
    adata : sc.AnnData
        Normalized AnnData object.
    time_key : str, default 'timepoint'
        Column in .obs containing timepoints.
    min_expressing_cells : int, default 10
        Minimum expressing cells required to compute regression slope.
    sub_col : str, optional
        Boolean column in .var for target genes (e.g. 'is_sub').
    tf_col : str, optional
        Boolean column in .var for transcription factors ('is_tf').

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'df_slopes', 'fig_boxplot', and 'fig_trend'.
    """
    # Input validation
    if not isinstance(adata, sc.AnnData):
        raise TypeError("Input 'adata' must be a scanpy AnnData object.")
    if time_key not in adata.obs.columns:
        raise KeyError(f"time_key '{time_key}' not found in adata.obs.")
    if min_expressing_cells < 1:
        raise ValueError("min_expressing_cells must be at least 1.")

    X: np.ndarray = (
        adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    )
    times: np.ndarray = _parse_time_values(adata.obs[time_key])
    genes: np.ndarray = adata.var_names.values

    # Compute linear regression for each gene
    results_list: List[Dict[str, Any]] = []
    for idx, g in enumerate(genes):
        exp: np.ndarray = X[:, idx]
        # Exclude genes with insufficient expression
        if (exp != 0).sum() < min_expressing_cells:
            results_list.append({
                "gene": g, "slope": np.nan, "pval": np.nan, "r_value": np.nan
            })
        else:
            reg = sci.linregress(times, exp)
            results_list.append({
                "gene": g, "slope": reg.slope,
                "pval": reg.pvalue
            })

    df_slopes: pd.DataFrame = pd.DataFrame(results_list).set_index("gene")
    # Classify genes
    df_slopes["cat"] = REST_NAME
    if tf_col and tf_col in adata.var.columns:
        df_slopes.loc[adata.var.index[adata.var[tf_col]], "cat"] = TF_NAME
    if sub_col and sub_col in adata.var.columns:
        df_slopes.loc[adata.var.index[adata.var[sub_col]], "cat"] = SUB_NAME

    # Boxplot of temporal slopes by category
    fig_box, ax_box = plt.subplots(figsize=(6, 5))
    cat_order: List[str] = [SUB_NAME, TF_NAME, REST_NAME]
    palette_box: Dict[str, str] = {
        SUB_NAME: COLORES["sig"],
        TF_NAME: COLORES["nosig"],
        REST_NAME: COLORES["bg"],
    }
    sns.boxplot(
        data=df_slopes.dropna(subset=["slope"]),
        x="cat",
        y="slope",
        hue="cat",
        legend=False,
        order=cat_order,
        palette=palette_box,
        showfliers=False,
        width=0.6,
        ax=ax_box,
    )
    ax_box.set_xlabel("")
    ax_box.set_ylabel("Temporal slope")
    ax_box.set_title("Temporal Expression Slope by Category")
    # Wrap x-tick labels to MAX_LABEL_CHARS characters per line
    wrapped_labels: List[str] = [
        textwrap.fill(cat, width=MAX_LABEL_CHARS) for cat in cat_order
    ]
    ax_box.set_xticks(range(len(cat_order)))
    ax_box.set_xticklabels(wrapped_labels)
    fig_box.tight_layout()
    # End plot

    # Plot of trend lines
    fig_trend, ax_trend = plt.subplots(figsize=(6, 5))
    medians: pd.Series = df_slopes.groupby("cat")["slope"].median()
    attr: List[Tuple[str, str, str, float]] = [
        (SUB_NAME, COLORES["sig"], "-", 2.0),
        (REST_NAME, COLORES["bg"], "--", 1.8),
        (TF_NAME, COLORES["nosig"], "--", 1.8),
    ]
    for cat, color, ls, lw in attr:
        m: float = float(medians.get(cat, np.nan))
        if pd.notna(m):
            ax_trend.axline(
                xy1=(0.5, 0),
                slope=m,
                color=color,
                linestyle=ls,
                linewidth=lw,
                label=f"{cat} = {m:.2e}",
            )

    ax_trend.set_xlabel("Time (Days)")
    ax_trend.set_ylabel("Expression trend")
    ax_trend.set_xlim(0, 1)
    ax_trend.xaxis.set_major_locator(ticker.MaxNLocator(1))
    ax_trend.yaxis.set_major_locator(ticker.MaxNLocator(2))

    med_max: Any = medians.abs().max()
    max_m: float = float(med_max) if pd.notna(med_max) else 0.0005
    y_lim: float = max(max_m, 0.0005)
    ax_trend.set_ylim(-y_lim, y_lim)
    ax_trend.legend(title="Median slopes")
    ax_trend.set_title("Comparison of Expression Trends by Category")
    fig_trend.tight_layout()

    # Close plots to prevent unwanted display
    plt.close(fig_box)
    plt.close(fig_trend)

    results: Dict[str, Any] = {
        "df_slopes": df_slopes,
        "fig_boxplot": fig_box,
        "fig_trend": fig_trend,
    }
    return results


# =============================================================================
#   MONTE CARLO PERMUTATION TESTS AND HYPOTHESIS TESTING
# =============================================================================

def run_tests(
    col_data: pd.Series,
    col_cat: pd.Series,
    sel_cat: Optional[str] = SUB_NAME,
    permutation_test: bool = True,
    mannwhitney_test: bool = True,
    n_permutations: int = 10000,
    random_state: Optional[int] = None,
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    """
    Execute a Monte Carlo permutation test comparing a target gene set
    against primary and secondary backgrounds.

    Parameters
    ----------
    col_data : pd.Series
        Numeric series indexed by gene identifiers.
    col_cat : pd.Series
        Categorical series indexed by gene identifiers.
    sel_cat : str
        Category to compare against the rest of the categories.
    permutation_test : bool, default True
        Whether to run the permutation test.
    mannwhitney_test : bool, default True
        Whether to run the Mann-Whitney test.
    n_permutations : int, default 10000
        Number of permutation iterations.
    random_state : int, default 99
        Seed for reproducibility.
    alternative : str, default 'two-sided'
        'less', 'greater', or 'two-sided'.

    Returns
    -------
    Dict[str, Any]
        Dictionary with observed statistic, permutation and Mann-Whitney
        p-values, null distributions, and figures.
    """
    # Input validation
    if not isinstance(col_data, pd.Series) or not isinstance(
        col_cat, pd.Series
    ):
        raise TypeError("col_data and col_cat must be pandas Series.")
    if col_data.dropna().empty:
        raise ValueError("col_data contains no valid numeric values.")
    if alternative not in {"less", "greater", "two-sided"}:
        raise ValueError(
            "alternative must be 'less', 'greater', or 'two-sided'."
        )
    if n_permutations < 1:
        raise ValueError("n_permutations must be greater than zero.")
    if (
        not col_cat.index.equals(col_data.index)
        or len(col_cat) < 1
        or sel_cat not in col_cat.unique()
    ):
        raise ValueError(
            "The index of col_cat must match the index of col_data, "
            "and sel_cat must be present in col_cat."
        )
    if not random_state: 
        random_state = np.random.randint(0,1000)
    np.random.seed(random_state)

    results: Dict[str, Any] = {}

    if permutation_test:
        # Median of the selected group and the rest of samples
        rst_median: float = float(np.median(col_data.loc[col_cat != sel_cat]))
        grp_median: float = float(np.median(col_data.loc[col_cat == sel_cat]))
        # Quantity of samples in the selected category
        n_target: int = int((col_cat == sel_cat).sum())

        # Preallocate array to store sub-sample medians
        null_dist: np.ndarray = np.empty(n_permutations)

        # Resampling: draw without replacement
        for i in range(n_permutations):
            rand_grp: np.ndarray = np.random.choice(
                col_data, size=n_target, replace=False
            )
            null_dist[i] = np.median(rand_grp)

        # Empirical p-value with correction
        if alternative == "less":
            p_perm: float = float(
                (np.sum(null_dist <= grp_median) + 1) / (n_permutations + 1)
            )
        elif alternative == "greater":
            p_perm = float(
                (np.sum(null_dist >= grp_median) + 1) / (n_permutations + 1)
            )
        else:  # two-sided
            diff_obs: float = float(np.abs(grp_median - rst_median))
            diff_null: np.ndarray = np.abs(null_dist - np.median(null_dist))
            p_perm = float(
                (np.sum(diff_null >= diff_obs) + 1) / (n_permutations + 1)
            )
        # End permutation test

        # Plot null permutation distribution
        fig_kde, ax_kde = plt.subplots(figsize=(6, 5))
        sns.kdeplot(
            null_dist, color=COLORES["bg"], fill=True, alpha=0.3,
            ax=ax_kde, label="Randomized permutations"
        )

        # Median of the selected group
        ax_kde.axvline(
            grp_median, color=COLORES["sig"], linewidth=2, linestyle="--",
            label=f"Selected group median: ({grp_median:.4e})"
        )

        # Median of the rest of samples
        ax_kde.axvline(
            rst_median, color=COLORES["nosig"], linewidth=2, linestyle="--",
            label=f"Rest of samples median: ({rst_median:.4e})"
        )

        # Plot parameters
        ax_kde.set_xlabel("Median distribution")
        ax_kde.set_ylabel("Density")
        ax_kde.set_title(
            f"Null Permutation Distribution\n"
            f"alternative={alternative}:(p-perm = {p_perm:.4e})"
        )
        ax_kde.legend()
        fig_kde.tight_layout()

        # Close plots to prevent unwanted display
        plt.close(fig_kde)
        # End plot

        results["grp_median"] = grp_median
        results["rst_median"] = rst_median
        results["p_perm"] = p_perm
        results["fig_dist"] = fig_kde

    if mannwhitney_test:
        # Wilcoxon test
        grp_data: pd.Series = col_data.loc[col_cat == sel_cat]
        rst_data: pd.Series = col_data.loc[col_cat != sel_cat]
        stat_val, p_w_val = sci.mannwhitneyu(
            grp_data, rst_data, alternative=alternative
        )
        # End Wilcoxon test

        results["p_mw"] = float(p_w_val)
        results["stat_mw"] = float(stat_val)

    return results


# =============================================================================
#   CELL-TYPE SLOPES AND Z-SCORES
# =============================================================================

def compute_celltype_slopes_and_zscores(
    adata: sc.AnnData,
    celltype_key: str = "cell_type",
    time_key: str = "timepoint",
    min_expressing_cells: int = 5,
    min_cells: int = 5,
    min_time: int = 2,
) -> Dict[str, Any]:
    """
    Compute regression slopes and Z-scores per cell type in-place.

    For each cell type, evaluates genes passing cell expression filters,
    computes linear regression slopes against time, and standardizes
    these slopes into Z-scores for each cell population.

    Parameters
    ----------
    adata : sc.AnnData
        Normalized AnnData object.
    celltype_key : str, default 'cell_type'
        Column in .obs specifying cell populations.
    time_key : str, default 'timepoint'
        Column in .obs specifying timepoints.
    min_expressing_cells : int, default 5
        Minimum expressing cells required for a gene in a cell type.
    min_cells : int, default 5
        Minimum cells required in a cell type.
    min_time : int, default 2
        Minimum number of unique timepoints in a cell type.

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'slopes' and 'z_scores' DataFrames.
    """
    # Input validation
    if not isinstance(adata, sc.AnnData):
        raise TypeError("Input 'adata' must be a scanpy AnnData object.")
    if celltype_key not in adata.obs.columns:
        raise KeyError(
            f"celltype_key '{celltype_key}' not found in adata.obs."
        )
    if time_key not in adata.obs.columns:
        raise KeyError(f"time_key '{time_key}' not found in adata.obs.")
    if min_expressing_cells < 0 or min_cells < 0 or min_time < 1:
        raise ValueError(
            "Threshold counts must be non-negative and min_time >= 1."
        )

    X: np.ndarray = (
        adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    )
    genes: np.ndarray = adata.var_names.values
    cell_types: List[str] = list(adata.obs[celltype_key].dropna().unique())
    times: np.ndarray = _parse_time_values(adata.obs[time_key])

    # Array preallocation to store slopes and z-scores
    slopes_mat: np.ndarray = np.full((len(genes), len(cell_types)), np.nan)
    z_scores_mat: np.ndarray = slopes_mat.copy()

    # Iterate through cell types and compute slopes
    for ct_idx, ct in enumerate(cell_types):
        cell_mask: pd.Series = adata.obs[celltype_key] == ct
        sub_time: np.ndarray = times[cell_mask]

        # Check cell types with few cells or few timepoints 
        if cell_mask.sum() < min_cells or len(np.unique(sub_time)) < min_time:
            continue

        # Check genes expression
        sub_X: np.ndarray = X[cell_mask, :]
        counts: np.ndarray = np.count_nonzero(sub_X, axis=0)
        valid_indices: np.ndarray = np.where(counts >= min_expressing_cells)[0]

        # Run linear regression for each gene
        for g_idx in valid_indices:
            res = sci.linregress(sub_time, sub_X[:, g_idx])
            slopes_mat[g_idx, ct_idx] = res.slope

    # Transform slopes into z-scores
    for ct_idx, ct in enumerate(cell_types):
        col_slopes: np.ndarray = slopes_mat[:, ct_idx]
        z_scores_mat[:, ct_idx] = sci.zscore(
            col_slopes, nan_policy="omit"
        )

    # Format the results
    df_slopes: pd.DataFrame = pd.DataFrame(
        slopes_mat, index=genes, columns=cell_types
    )
    df_zscores: pd.DataFrame = pd.DataFrame(
        z_scores_mat, index=genes, columns=cell_types
    )

    results: Dict[str, Any] = {
        "slopes": df_slopes,
        "z_scores": df_zscores,
    }
    return results


# =============================================================================
#   HEATMAP WITH HIGHLIGHTED CELL IDENTITY MARKERS
# =============================================================================

def plot_heatmap_with_highlights(
    df: pd.DataFrame,
    celltype_gene_map: Optional[Dict[str, Set[str]]] = None,
    subset_genes: Optional[Union[Set[str], List[str]]] = None,
    vmin: float = -3.0,
    vmax: float = 3.0,
    cmap: str = "coolwarm",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 20),
) -> Dict[str, Any]:
    """
    Render a heatmap bounded in [vmin, vmax] with black rectangle patches
    highlighting cell-type specific markers.

    Replicates the exact heatmap visualization from anexos.qmd (line 1798).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a matrix of numbers (genes x cells).
    celltype_gene_map : dict, optional
        Dictionary {cell_type: set(specific_markers)}.
    subset_genes : set or list of str, optional
        Subset of genes to include in the heatmap.
    vmin, vmax : float, default -3.0, 3.0
        Color scale minimum and maximum bounds.
    cmap : str, default 'coolwarm'
        Colormap name.
    title : str, optional
        Figure title.
    figsize : tuple, default (20, 20)
        Figure dimensions.

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'fig'.
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input 'df' cannot be empty.")
    if vmin >= vmax:
        raise ValueError("vmin must be strictly less than vmax.")

    # Plot heatmap with highlighted patches (heat-trend)
    clean_df: pd.DataFrame = df.copy()

    # Subset expression matrix to target genes if specified
    subset_genes_set: Set[str] = (
        set(subset_genes) if subset_genes is not None else set()
    )
    if len(subset_genes_set) > 0:
        subset_genes_set = subset_genes_set.intersection(clean_df.index)
        clean_df = clean_df.loc[list(subset_genes_set)]

    clean_df = clean_df.sort_index()

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#FFFFFF")

    # Wrap tick labels to MAX_LABEL_CHARS characters per line
    wrapped_x: List[str] = [
        textwrap.fill(str(col), width=MAX_LABEL_CHARS)
        for col in clean_df.columns
    ]
    wrapped_y: List[str] = [
        textwrap.fill(str(row), width=MAX_LABEL_CHARS)
        for row in clean_df.index
    ]

    # Plot heatmap
    sns.heatmap(
        clean_df,
        cmap=cmap,
        center=0,
        vmin=vmin,
        vmax=vmax,
        linewidths=1,
        annot=False,
        cbar=True,
        cbar_kws={
            "shrink": 0.5
        },
        square=True,
        ax=ax,
    )

    # Frame cell type specific Terminal Selector markers with black borders
    if celltype_gene_map:
        cells_to_highlight: Set[Tuple[int, int]] = set()
        for i, neu in enumerate(clean_df.columns):
            if neu in celltype_gene_map:
                neu_sub: Set[str] = celltype_gene_map[neu]
                for j, gene in enumerate(clean_df.index):
                    if gene in neu_sub:
                        cells_to_highlight.add((j, i))

        for row_idx, col_idx in cells_to_highlight:
            ax.add_patch(
                patches.Rectangle(
                    (col_idx, row_idx), 1, 1,
                    fill=False,
                    edgecolor="black",
                    lw=2,
                    clip_on=False,
                )
            )

    # Plot typography and colorbar attributes
    if ax.collections and ax.collections[0].colorbar:
        ax.collections[0].colorbar.ax.tick_params(labelsize=30)
    plt.xticks(fontsize=10, fontweight="bold", rotation=90)
    plt.ylabel("")
    plt.yticks(fontsize=10, fontweight="bold", rotation=0)
    if title:
        ax.set_title(title, fontsize=20, pad=25)

    fig.tight_layout()

    # Close plots to prevent unwanted display
    plt.close(fig)

    results: Dict[str, Any] = {
        "fig": fig,
    }
    return results