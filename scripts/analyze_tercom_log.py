#!/usr/bin/env python3
"""TERCOM Navigation Log Analyzer.

Reads a CSV produced by diagnostics_node and generates publication-quality
figures for algorithmic analysis of TERCOM GPS-denied navigation performance.

Usage:
    python3 analyze_tercom_log.py <path/to/tercom_log.csv> [--outdir <dir>]

Output figures (saved as PNG + PDF by default):
  01_trajectory_xy          - XY ground track: estimated vs ground truth
  02_position_error_time    - Per-axis and horizontal/3D error vs time
  03_error_statistics       - Sliding-window RMSE, mean, max horizontal error
  04_covariance_consistency - 3-sigma position bounds vs actual error
  05_nis_time               - NIS vs time with chi-squared consistency bounds
  06_tercom_quality         - MAD, discrimination, roughness, noise vs time
  07_speed_profile          - Estimated and true speed vs time
  08_filter_state_timeline  - Color-coded filter state over time
  09_error_histogram        - Distribution of horizontal and vertical errors
  10_cov_vs_error           - Covariance sigma vs actual error (consistency scatter)
  11_tercom_mad_vs_error    - TERCOM MAD vs position error scatter
  12_trajectory_3d          - 3D trajectory comparison
  13_accepted_fixes_rate    - Cumulative accepted TERCOM fixes vs time
  14_health_metrics         - Filter health: max_pos_std, innov_norm, is_healthy
  15_summary_dashboard      - Single-page multi-panel summary figure
"""
import argparse
import functools
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from scipy.stats import chi2

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 120,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'lines.linewidth': 1.4,
})

# Module-level output format list — overridden by main() from CLI args
OUTPUT_FORMATS: list = ['png']

# ── Filter-state color palette ────────────────────────────────────────────────
STATE_COLORS = {
    'WAITING_GPS':  '#aaaaaa',
    'INITIALIZING': '#f9c74f',
    'RUNNING':      '#90be6d',
    'DIVERGED':     '#f94144',
    'RESETTING':    '#f3722c',
    'UNKNOWN':      '#cccccc',
}

CHI2_DOF2_95 = chi2.ppf(0.95, df=2)  # 2-DOF NIS upper bound ≈ 5.99
CHI2_DOF2_05 = chi2.ppf(0.05, df=2)  # 2-DOF NIS lower bound ≈ 0.10


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalise time to seconds from first row
    if 'time_s' not in df.columns and 'ros_timestamp_ns' in df.columns:
        df['time_s'] = (df['ros_timestamp_ns'] - df['ros_timestamp_ns'].iloc[0]) * 1e-9

    # ── Frame-aligned ground truth ─────────────────────────────────────────────
    # The CSV stores raw MAVROS ground truth (true_x/y/z) in PX4's local NED boot
    # frame and ESKF estimates (est_x/y/z) in a UTM-derived ENU frame.  These two
    # frames differ by a constant offset that diagnostics_node computes once when
    # the filter first enters RUNNING state and then applies consistently when
    # writing err_x/y/z = est_x/y/z - (true_x/y/z + offset_x/y/z).
    #
    # We can therefore reconstruct the frame-aligned ground truth from the columns
    # already in the CSV without knowing the offset explicitly:
    #   true_x_aligned = est_x - err_x
    #
    # All trajectory plots MUST use true_x/y/z_aligned, NOT true_x/y/z, to avoid
    # the appearance of hundreds-of-metres drift caused purely by the frame offset.
    if all(c in df.columns for c in ['est_x', 'est_y', 'est_z',
                                      'err_x', 'err_y', 'err_z']):
        df['true_x_aligned'] = df['est_x'] - df['err_x']
        df['true_y_aligned'] = df['est_y'] - df['err_y']
        df['true_z_aligned'] = df['est_z'] - df['err_z']

    return df


def save_fig(fig: plt.Figure, outdir: str, name: str):
    for fmt in OUTPUT_FORMATS:
        path = os.path.join(outdir, f'{name}.{fmt}')
        fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def state_segments(df: pd.DataFrame):
    """Return list of (t_start, t_end, state) for coloring background bands."""
    segs = []
    if 'filter_state' not in df.columns:
        return segs
    t = df['time_s'].values
    s = df['filter_state'].values
    i0, cur = 0, s[0]
    for i in range(1, len(s)):
        if s[i] != cur:
            segs.append((t[i0], t[i], cur))
            i0, cur = i, s[i]
    segs.append((t[i0], t[-1], cur))
    return segs


def add_state_background(ax, segs, alpha=0.12):
    for t0, t1, st in segs:
        ax.axvspan(t0, t1, color=STATE_COLORS.get(st, '#cccccc'), alpha=alpha, lw=0)


def running_mask(df: pd.DataFrame) -> np.ndarray:
    if 'filter_state' not in df.columns:
        return np.ones(len(df), dtype=bool)
    return (df['filter_state'] == 'RUNNING').values


# ─────────────────────────────────────────────────────────────────────────────
# Figure generators
# ─────────────────────────────────────────────────────────────────────────────

def fig_trajectory_xy(df: pd.DataFrame, outdir: str):
    fig, ax = plt.subplots(figsize=(8, 7))

    # Use frame-aligned ground truth so both trajectories are in the same ENU frame.
    # true_x/y in the CSV are raw MAVROS positions (PX4 boot frame); plotting them
    # directly against est_x/y would show a spurious constant offset of potentially
    # hundreds of metres.  true_x_aligned = est_x - err_x is always correct.
    tx = df['true_x_aligned'] if 'true_x_aligned' in df.columns else df['true_x']
    ty = df['true_y_aligned'] if 'true_y_aligned' in df.columns else df['true_y']
    using_raw = 'true_x_aligned' not in df.columns

    ax.plot(tx, ty, color='#2196F3', lw=1.6,
            label='Ground Truth (frame-aligned)', zorder=2)
    ax.plot(df['est_x'], df['est_y'], color='#FF5722', lw=1.2, ls='--',
            label='ESKF Estimate', zorder=3)

    # Mark start and end
    ax.scatter(tx.iloc[0], ty.iloc[0], marker='o', s=80,
               color='green', zorder=5, label='Start')
    ax.scatter(tx.iloc[-1], ty.iloc[-1], marker='s', s=80,
               color='red', zorder=5, label='End')

    # Mark accepted TERCOM fixes if count changes
    if 'tercom_accepted_count' in df.columns:
        fix_rows = df[df['tercom_accepted_count'].diff().fillna(0) > 0]
        ax.scatter(fix_rows['est_x'], fix_rows['est_y'], marker='+', s=60,
                   color='lime', linewidths=1.5, zorder=4, label='TERCOM Fix Applied')

    # Annotate the frame offset so the reader understands the coordinate alignment
    if 'true_x_aligned' in df.columns and 'true_x' in df.columns:
        off_x = float((df['true_x_aligned'] - df['true_x']).mean())
        off_y = float((df['true_y_aligned'] - df['true_y']).mean())
        off_norm = np.hypot(off_x, off_y)
        off_txt = (f'Frame offset (MAVROS→ENU):\n'
                   f'  ΔE={off_x:.1f} m, ΔN={off_y:.1f} m\n'
                   f'  ‖offset‖={off_norm:.1f} m')
        ax.text(0.02, 0.02, off_txt, transform=ax.transAxes, fontsize=8,
                va='bottom', color='#555',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    if using_raw:
        ax.text(0.5, 0.5, 'WARNING: err_x/y columns missing;\nusing raw true_x/y (frames may differ)',
                transform=ax.transAxes, ha='center', va='center',
                color='red', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title('XY Ground Track: ESKF Estimate vs Ground Truth\n'
                 '(ground truth shifted by frame alignment offset into ENU frame)')
    ax.legend(loc='best')
    ax.set_aspect('equal', adjustable='datalim')
    fig.tight_layout()
    save_fig(fig, outdir, '01_trajectory_xy')


def fig_trajectory_satellite(df: pd.DataFrame, outdir: str,
                              origin_lat: float = None, origin_lon: float = None,
                              satellite_zoom: int = 17):
    """2D trajectory plot over a satellite basemap.

    Requires --origin-lat and --origin-lon (the geographic coordinates of the
    ENU frame origin, i.e. world_origin_lat / world_origin_lon in the launch
    params).  The function converts ENU (x=East, y=North) metre coordinates
    into Web Mercator (EPSG:3857) so that contextily can overlay satellite tiles.

    Tile source: Esri World Imagery (no API key required).
    Falls back gracefully if contextily / pyproj are unavailable or origin
    coordinates are not supplied.
    """
    if origin_lat is None or origin_lon is None:
        warnings.warn(
            'fig_trajectory_satellite: skipped — pass --origin-lat and '
            '--origin-lon to enable the satellite-map trajectory plot.')
        return

    try:
        import contextily as ctx
        from pyproj import Transformer
    except ImportError as exc:
        warnings.warn(f'fig_trajectory_satellite: skipped — {exc}')
        return

    # ── Coordinate conversion: ENU (m) → Web Mercator (m) ──────────────────
    # Auto-detect UTM zone from the origin's longitude.
    utm_zone = int((origin_lon + 180.0) / 6.0) + 1
    utm_epsg = 32600 + utm_zone  # North hemisphere (32700 + zone for South)
    if origin_lat < 0:
        utm_epsg = 32700 + utm_zone

    # Geographic origin → UTM easting/northing
    tf_geo_to_utm = Transformer.from_crs('EPSG:4326', f'EPSG:{utm_epsg}',
                                          always_xy=True)
    origin_e, origin_n = tf_geo_to_utm.transform(origin_lon, origin_lat)

    # UTM → Web Mercator
    tf_utm_to_web = Transformer.from_crs(f'EPSG:{utm_epsg}', 'EPSG:3857',
                                          always_xy=True)

    def enu_to_web(x_enu, y_enu):
        utm_e = origin_e + x_enu
        utm_n = origin_n + y_enu
        return tf_utm_to_web.transform(utm_e, utm_n)

    # Ground truth trajectory (frame-aligned)
    tx = df['true_x_aligned'] if 'true_x_aligned' in df.columns else df['true_x']
    ty = df['true_y_aligned'] if 'true_y_aligned' in df.columns else df['true_y']

    gt_wx, gt_wy   = enu_to_web(tx.values, ty.values)
    est_wx, est_wy = enu_to_web(df['est_x'].values, df['est_y'].values)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8), dpi=200)

    ax.plot(gt_wx, gt_wy, color='#2196F3', lw=2.0,
            label='Ground Truth (frame-aligned)', zorder=3)
    ax.plot(est_wx, est_wy, color='#FF5722', lw=1.4, ls='--',
            label='ESKF Estimate', zorder=4)

    # Start / end markers
    ax.scatter(gt_wx[0], gt_wy[0], marker='o', s=100,
               color='lime', edgecolors='k', lw=0.8, zorder=6, label='Start')
    ax.scatter(gt_wx[-1], gt_wy[-1], marker='s', s=100,
               color='red', edgecolors='k', lw=0.8, zorder=6, label='End')

    # TERCOM fix locations
    if 'tercom_accepted_count' in df.columns:
        fix_rows = df[df['tercom_accepted_count'].diff().fillna(0) > 0]
        if not fix_rows.empty:
            fx, fy = enu_to_web(fix_rows['est_x'].values,
                                 fix_rows['est_y'].values)
            ax.scatter(fx, fy, marker='+', s=70, color='yellow',
                       linewidths=1.8, zorder=5, label='TERCOM Fix Applied')

    # ── Satellite basemap ─────────────────────────────────────────────────────
    try:
        ctx.add_basemap(
            ax,
            crs='EPSG:3857',
            source=ctx.providers.Esri.WorldImagery,
            zoom=satellite_zoom,
            attribution_size=7,
        )
    except Exception as tile_exc:
        ax.text(0.5, 0.5,
                f'Satellite tiles unavailable:\n{tile_exc}',
                transform=ax.transAxes, ha='center', va='center',
                color='red', fontsize=9,
                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

    ax.set_xlabel('Web Mercator X (m)')
    ax.set_ylabel('Web Mercator Y (m)')
    ax.set_title('XY Ground Track over Satellite Map\n'
                 f'Origin: {origin_lat:.6f}°N, {origin_lon:.6f}°E  '
                 f'(UTM zone {utm_zone}{"N" if origin_lat >= 0 else "S"})')
    ax.legend(loc='best', framealpha=0.85)
    fig.tight_layout()
    save_fig(fig, outdir, '01b_trajectory_satellite')


def fig_position_error_time(df: pd.DataFrame, outdir: str):
    segs = state_segments(df)
    t = df['time_s'].values

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Per-axis errors
    for ax, (col, label, color) in zip(axes[:2], [
        ('err_h_norm', 'Horizontal Error ‖Δh‖ (m)',  '#E91E63'),
        ('err_v_abs',  'Vertical Error |Δz| (m)',     '#9C27B0'),
    ]):
        add_state_background(ax, segs)
        ax.plot(t, df[col], color=color, lw=1.2)
        ax.set_ylabel(label)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # X/Y/Z breakdown
    ax = axes[2]
    add_state_background(ax, segs)
    ax.plot(t, df['err_x'], color='#F44336', lw=1.0, label='ΔX (East)')
    ax.plot(t, df['err_y'], color='#4CAF50', lw=1.0, label='ΔY (North)')
    ax.plot(t, df['err_z'], color='#2196F3', lw=1.0, label='ΔZ (Up)')
    ax.axhline(0, color='k', lw=0.6, ls=':')
    ax.set_ylabel('Per-Axis Error (m)')
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right', ncol=3)

    # Legend for filter states
    patches = [mpatches.Patch(color=c, alpha=0.4, label=s)
               for s, c in STATE_COLORS.items() if any(sg[2] == s for sg in segs)]
    if patches:
        axes[0].legend(handles=patches, loc='upper right', fontsize=9,
                       title='Filter State', framealpha=0.7)

    axes[0].set_title('Position Error vs Time')
    fig.tight_layout()
    save_fig(fig, outdir, '02_position_error_time')


def fig_error_statistics(df: pd.DataFrame, outdir: str):
    t = df['time_s'].values
    h = df['err_h_norm'].values
    segs = state_segments(df)

    # Sliding-window statistics (window = 5 s → adapt to sample rate)
    dt_avg = (t[-1] - t[0]) / max(len(t) - 1, 1)
    win = max(10, int(5.0 / dt_avg))

    rms  = np.array([np.sqrt(np.mean(h[max(0, i-win):i+1]**2))
                     for i in range(len(h))])
    mean = np.array([np.mean(h[max(0, i-win):i+1]) for i in range(len(h))])
    mx   = np.array([np.max(h[max(0, i-win):i+1])  for i in range(len(h))])

    fig, ax = plt.subplots(figsize=(12, 5))
    add_state_background(ax, segs)
    ax.fill_between(t, 0, mx,   alpha=0.15, color='#F44336', label='Sliding Max')
    ax.plot(t, mx,   color='#F44336', lw=0.8, ls=':')
    ax.plot(t, rms,  color='#FF9800', lw=1.5, label=f'Sliding RMS ({win}-sample window)')
    ax.plot(t, mean, color='#4CAF50', lw=1.5, label='Sliding Mean')
    ax.plot(t, h,    color='#9E9E9E', lw=0.6, alpha=0.6, label='Instantaneous')

    # Global statistics annotation
    mask = running_mask(df)
    if mask.any():
        h_run = h[mask]
        txt = (f'RUNNING phase — RMS: {np.sqrt(np.mean(h_run**2)):.2f} m  '
               f'Mean: {np.mean(h_run):.2f} m  Max: {np.max(h_run):.2f} m  '
               f'P95: {np.percentile(h_run, 95):.2f} m')
        ax.set_title(f'Horizontal Error Statistics\n{txt}')
    else:
        ax.set_title('Horizontal Error Statistics')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Horizontal Error (m)')
    ax.legend(loc='upper right')
    fig.tight_layout()
    save_fig(fig, outdir, '03_error_statistics')


def fig_covariance_consistency(df: pd.DataFrame, outdir: str):
    t = df['time_s'].values
    segs = state_segments(df)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for ax, (err_col, cov_col, axis_label) in zip(axes, [
        ('err_x', 'cov_xx', 'X (East)'),
        ('err_y', 'cov_yy', 'Y (North)'),
        ('err_z', 'cov_zz', 'Z (Up)'),
    ]):
        add_state_background(ax, segs)
        sigma = np.sqrt(np.abs(df[cov_col].values))
        ax.fill_between(t, -3*sigma, 3*sigma, alpha=0.2, color='#2196F3',
                        label='±3σ ESKF bound')
        ax.plot(t, df[err_col].values, color='#F44336', lw=1.0,
                label=f'Error {axis_label}')
        ax.axhline(0, color='k', lw=0.5, ls=':')
        ax.set_ylabel(f'{axis_label} Error (m)')
        ax.legend(loc='upper right')

    axes[0].set_title('Covariance Consistency: Actual Error vs ESKF ±3σ Bounds')
    axes[2].set_xlabel('Time (s)')
    fig.tight_layout()
    save_fig(fig, outdir, '04_covariance_consistency')


def fig_nis_time(df: pd.DataFrame, outdir: str):
    if 'nis' not in df.columns:
        return
    t = df['time_s'].values
    nis = df['nis'].values
    segs = state_segments(df)

    fig, ax = plt.subplots(figsize=(12, 5))
    add_state_background(ax, segs)
    ax.plot(t, nis, color='#673AB7', lw=1.2, label='NIS')
    ax.axhline(CHI2_DOF2_95, color='#F44336', lw=1.2, ls='--',
               label=f'χ²(2, 0.95) = {CHI2_DOF2_95:.2f}')
    ax.axhline(CHI2_DOF2_05, color='#4CAF50', lw=1.2, ls='--',
               label=f'χ²(2, 0.05) = {CHI2_DOF2_05:.2f}')
    ax.fill_between(t, CHI2_DOF2_05, CHI2_DOF2_95, alpha=0.1, color='#4CAF50',
                    label='Consistent (5%–95% band)')

    mask = running_mask(df) & (nis > 0)
    if mask.sum() > 5:
        frac_ok = np.mean((nis[mask] >= CHI2_DOF2_05) & (nis[mask] <= CHI2_DOF2_95))
        ax.set_title(f'Normalized Innovation Squared (NIS) vs Time\n'
                     f'Consistency (RUNNING): {frac_ok*100:.1f}% samples within bounds')
    else:
        ax.set_title('Normalized Innovation Squared (NIS) vs Time')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('NIS')
    ax.set_ylim(bottom=0, top=min(np.percentile(nis[nis > 0], 99) * 1.3
                                  if (nis > 0).any() else 20, 30))
    ax.legend(loc='upper right')
    fig.tight_layout()
    save_fig(fig, outdir, '05_nis_time')


def fig_tercom_quality(df: pd.DataFrame, outdir: str):
    t = df['time_s'].values
    segs = state_segments(df)

    cols = {
        'tercom_mad':        ('MAD (m)', '#E91E63',   None),
        'tercom_disc':       ('Discrimination', '#FF9800', 1.02),
        'tercom_roughness':  ('Roughness (m)', '#4CAF50', 5.0),
        'tercom_noise':      ('Noise (m)',  '#9C27B0',   None),
    }
    available = [(k, v) for k, v in cols.items() if k in df.columns]
    if not available:
        return

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (col, (ylabel, color, threshold)) in zip(axes, available):
        add_state_background(ax, segs)
        ax.plot(t, df[col], color=color, lw=1.0, label=col)
        if threshold is not None:
            ax.axhline(threshold, color='k', lw=0.9, ls='--',
                       label=f'Rejection threshold = {threshold}')
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right')

    axes[0].set_title('TERCOM Match Quality Metrics vs Time')
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    save_fig(fig, outdir, '06_tercom_quality')


def fig_speed_profile(df: pd.DataFrame, outdir: str):
    t = df['time_s'].values
    segs = state_segments(df)

    fig, ax = plt.subplots(figsize=(12, 4))
    add_state_background(ax, segs)
    if 'true_speed' in df.columns:
        ax.plot(t, df['true_speed'], color='#2196F3', lw=1.4, label='True Speed')
    if 'est_speed' in df.columns:
        ax.plot(t, df['est_speed'], color='#FF5722', lw=1.0, ls='--',
                label='Estimated Speed')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Vehicle Speed Profile')
    ax.legend(loc='upper right')
    fig.tight_layout()
    save_fig(fig, outdir, '07_speed_profile')


def fig_filter_state_timeline(df: pd.DataFrame, outdir: str):
    if 'filter_state' not in df.columns:
        return
    t = df['time_s'].values
    states = ['WAITING_GPS', 'INITIALIZING', 'RUNNING', 'DIVERGED', 'RESETTING', 'UNKNOWN']

    fig, ax = plt.subplots(figsize=(12, 3))
    segs = state_segments(df)
    for t0, t1, st in segs:
        ax.barh(0, t1 - t0, left=t0, height=0.8,
                color=STATE_COLORS.get(st, '#cccccc'), label=st)
        ax.text((t0 + t1) / 2, 0, st, ha='center', va='center',
                fontsize=8, fontweight='bold',
                color='white' if st in ('RUNNING', 'DIVERGED', 'RESETTING') else 'black')

    # Duration breakdown
    lines = []
    for st in states:
        dur = sum(t1 - t0 for t0, t1, s in segs if s == st)
        if dur > 0:
            lines.append(f'{st}: {dur:.1f}s ({dur/(t[-1]-t[0])*100:.1f}%)')
    ax.set_title('Filter State Timeline\n' + '   '.join(lines))
    ax.set_xlabel('Time (s)')
    ax.set_yticks([])
    ax.set_xlim(t[0], t[-1])
    fig.tight_layout()
    save_fig(fig, outdir, '08_filter_state_timeline')


def fig_error_histogram(df: pd.DataFrame, outdir: str):
    mask = running_mask(df)
    h_err = df['err_h_norm'].values[mask]
    v_err = df['err_v_abs'].values[mask] if 'err_v_abs' in df.columns else None

    n_plots = 2 if v_err is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, data, label, color in zip(
        axes,
        [h_err] + ([v_err] if v_err is not None else []),
        ['Horizontal Error (m)', 'Vertical Error |ΔZ| (m)'],
        ['#E91E63', '#9C27B0'],
    ):
        if len(data) == 0:
            continue
        ax.hist(data, bins=50, color=color, alpha=0.75, edgecolor='white', lw=0.4)
        rms = np.sqrt(np.mean(data**2))
        ax.axvline(rms, color='k', lw=1.5, ls='--', label=f'RMS = {rms:.2f} m')
        ax.axvline(np.mean(data), color='navy', lw=1.2, ls=':',
                   label=f'Mean = {np.mean(data):.2f} m')
        ax.axvline(np.percentile(data, 95), color='darkorange', lw=1.2, ls='-.',
                   label=f'P95 = {np.percentile(data, 95):.2f} m')
        ax.set_xlabel(label)
        ax.set_ylabel('Count')
        ax.set_title(f'{label} Distribution\n(RUNNING phase, N={len(data)})')
        ax.legend()

    fig.tight_layout()
    save_fig(fig, outdir, '09_error_histogram')


def fig_cov_vs_error(df: pd.DataFrame, outdir: str):
    mask = running_mask(df)
    if not mask.any():
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (err_col, cov_col, label) in zip(axes, [
        ('err_x', 'cov_xx', 'X (East)'),
        ('err_y', 'cov_yy', 'Y (North)'),
        ('err_z', 'cov_zz', 'Z (Up)'),
    ]):
        sigma = np.sqrt(np.abs(df[cov_col].values[mask]))
        err   = np.abs(df[err_col].values[mask])
        lim = max(np.percentile(sigma, 99), np.percentile(err, 99)) * 1.05
        ax.scatter(sigma, err, s=4, alpha=0.3, color='#673AB7')
        ax.plot([0, lim], [0, lim],   color='k',        lw=1.0, ls='--', label='1σ')
        ax.plot([0, lim], [0, 2*lim], color='#F44336',  lw=1.0, ls=':',  label='2σ')
        ax.plot([0, lim], [0, 3*lim], color='#FF9800',  lw=1.0, ls='-',  label='3σ')
        ax.set_xlabel(f'ESKF σ_{label[0].lower()} (m)')
        ax.set_ylabel(f'|Error_{label[0]}| (m)')
        ax.set_title(f'Covariance Consistency — {label}')
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.legend(fontsize=9)

    fig.suptitle('Covariance vs Absolute Error (RUNNING phase)', fontsize=14, y=1.01)
    fig.tight_layout()
    save_fig(fig, outdir, '10_cov_vs_error')


def fig_tercom_mad_vs_error(df: pd.DataFrame, outdir: str):
    if 'tercom_mad' not in df.columns:
        return
    mask = running_mask(df) & (df['tercom_mad'].values > 0)
    if not mask.any():
        return

    mad  = df['tercom_mad'].values[mask]
    herr = df['err_h_norm'].values[mask]

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(mad, herr, s=8, c=herr, cmap='plasma', alpha=0.5)
    plt.colorbar(sc, ax=ax, label='Horizontal Error (m)')

    # Trend line
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        coeffs = np.polyfit(mad, herr, 1)
    trend = np.poly1d(coeffs)
    xs = np.linspace(mad.min(), mad.max(), 200)
    ax.plot(xs, trend(xs), color='k', lw=1.5, ls='--',
            label=f'Linear fit: err = {coeffs[0]:.2f}·MAD + {coeffs[1]:.2f}')

    ax.set_xlabel('TERCOM MAD (m)')
    ax.set_ylabel('Horizontal Position Error (m)')
    ax.set_title('TERCOM Match Quality (MAD) vs Position Error\n'
                 '(RUNNING phase — lower MAD should correlate with lower error)')
    ax.axvline(30.0, color='r', lw=1, ls=':', label='MAD rejection threshold (30 m)')
    ax.legend()
    fig.tight_layout()
    save_fig(fig, outdir, '11_tercom_mad_vs_error')


def fig_trajectory_3d(df: pd.DataFrame, outdir: str):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    tx = df['true_x_aligned'] if 'true_x_aligned' in df.columns else df['true_x']
    ty = df['true_y_aligned'] if 'true_y_aligned' in df.columns else df['true_y']
    tz = df['true_z_aligned'] if 'true_z_aligned' in df.columns else df['true_z']

    ax.plot(tx, ty, tz,
            color='#2196F3', lw=1.4, label='Ground Truth (frame-aligned)', zorder=2)
    ax.plot(df['est_x'],  df['est_y'],  df['est_z'],
            color='#FF5722', lw=1.0, ls='--', label='ESKF Estimate', zorder=3)

    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Up (m)')
    ax.set_title('3D Trajectory: ESKF Estimate vs Ground Truth\n'
                 '(ground truth shifted into ENU frame by alignment offset)')
    ax.legend()
    fig.tight_layout()
    save_fig(fig, outdir, '12_trajectory_3d')


def fig_accepted_fixes_rate(df: pd.DataFrame, outdir: str):
    if 'tercom_accepted_count' not in df.columns:
        return
    t = df['time_s'].values
    count = df['tercom_accepted_count'].values

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.step(t, count, where='post', color='#4CAF50', lw=1.5,
            label='Cumulative Accepted Fixes')
    ax.fill_between(t, 0, count, step='post', alpha=0.2, color='#4CAF50')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative TERCOM Fixes Accepted')
    ax.set_title(f'Cumulative Accepted TERCOM Fixes vs Time\n'
                 f'Total accepted: {int(count[-1])}')
    ax.legend(loc='upper left')

    # Instantaneous fix rate annotation
    dt = t[-1] - t[0]
    if dt > 0:
        rate = count[-1] / dt
        ax.text(0.98, 0.95, f'Avg rate: {rate:.3f} Hz',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    fig.tight_layout()
    save_fig(fig, outdir, '13_accepted_fixes_rate')


def fig_health_metrics(df: pd.DataFrame, outdir: str):
    cols_to_plot = [
        ('health_max_pos_std', 'Max Position σ (m)', '#E91E63'),
        ('health_innov_norm',  'Innovation Norm',    '#FF9800'),
        ('health_is_healthy',  'Is Healthy (1=Yes)', '#4CAF50'),
    ]
    available = [(c, l, co) for c, l, co in cols_to_plot if c in df.columns]
    if not available:
        return

    t = df['time_s'].values
    segs = state_segments(df)

    fig, axes = plt.subplots(len(available), 1, figsize=(12, 3.5 * len(available)),
                             sharex=True)
    if len(available) == 1:
        axes = [axes]

    for ax, (col, label, color) in zip(axes, available):
        add_state_background(ax, segs)
        ax.plot(t, df[col], color=color, lw=1.0)
        ax.set_ylabel(label)

    axes[0].set_title('Filter Health Metrics vs Time')
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    save_fig(fig, outdir, '14_health_metrics')


def fig_summary_dashboard(df: pd.DataFrame, outdir: str):
    """Single-page 6-panel summary suitable for a paper appendix."""
    mask = running_mask(df)
    t = df['time_s'].values
    segs = state_segments(df)

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

    # ── Panel 1: XY Trajectory ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    tx1 = df['true_x_aligned'] if 'true_x_aligned' in df.columns else df['true_x']
    ty1 = df['true_y_aligned'] if 'true_y_aligned' in df.columns else df['true_y']
    ax1.plot(tx1, ty1, color='#2196F3', lw=1.2, label='GT (aligned)')
    ax1.plot(df['est_x'],  df['est_y'],  color='#FF5722', lw=0.9, ls='--', label='ESKF')
    ax1.set_xlabel('East (m)'); ax1.set_ylabel('North (m)')
    ax1.set_title('XY Ground Track'); ax1.legend(fontsize=9)
    ax1.set_aspect('equal', adjustable='datalim')

    # ── Panel 2: Horizontal Error vs Time ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1:])
    add_state_background(ax2, segs)
    ax2.plot(t, df['err_h_norm'], color='#E91E63', lw=1.0, label='Horizontal Error')
    if 'err_v_abs' in df.columns:
        ax2.plot(t, df['err_v_abs'], color='#9C27B0', lw=1.0, ls='--',
                 label='|Vertical Error|')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Error (m)')
    ax2.set_title('Position Error vs Time'); ax2.legend(fontsize=9)

    # ── Panel 3: NIS Consistency ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0:2])
    if 'nis' in df.columns:
        add_state_background(ax3, segs)
        ax3.plot(t, df['nis'], color='#673AB7', lw=0.9, label='NIS')
        ax3.axhline(CHI2_DOF2_95, color='r', lw=1, ls='--',
                    label=f'χ²(0.95)={CHI2_DOF2_95:.1f}')
        ax3.axhline(CHI2_DOF2_05, color='g', lw=1, ls='--',
                    label=f'χ²(0.05)={CHI2_DOF2_05:.2f}')
        ax3.set_ylim(bottom=0)
    ax3.set_xlabel('Time (s)'); ax3.set_ylabel('NIS')
    ax3.set_title('NIS Consistency'); ax3.legend(fontsize=8)

    # ── Panel 4: Error Histogram ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    if mask.any():
        h_err = df['err_h_norm'].values[mask]
        ax4.hist(h_err, bins=40, color='#E91E63', alpha=0.7, edgecolor='white', lw=0.3)
        rms = np.sqrt(np.mean(h_err**2))
        ax4.axvline(rms, color='k', lw=1.2, ls='--', label=f'RMS={rms:.1f}m')
        ax4.axvline(np.percentile(h_err, 95), color='orange', lw=1.2, ls='-.',
                    label=f'P95={np.percentile(h_err, 95):.1f}m')
        ax4.set_xlabel('Horizontal Error (m)'); ax4.set_ylabel('Count')
        ax4.set_title('Horizontal Error Dist.\n(RUNNING phase)')
        ax4.legend(fontsize=8)

    # ── Panel 5: TERCOM MAD vs Time ──────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0:2])
    if 'tercom_mad' in df.columns:
        add_state_background(ax5, segs)
        ax5.plot(t, df['tercom_mad'], color='#FF9800', lw=1.0, label='MAD')
        ax5.axhline(30.0, color='r', lw=0.9, ls='--', label='Reject > 30 m')
        ax5.set_ylim(bottom=0)
    ax5.set_xlabel('Time (s)'); ax5.set_ylabel('MAD (m)')
    ax5.set_title('TERCOM Match Quality (MAD)'); ax5.legend(fontsize=9)

    # ── Panel 6: Covariance σ_x vs |err_x| ──────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    if mask.any():
        sx  = np.sqrt(np.abs(df['cov_xx'].values[mask]))
        ex  = np.abs(df['err_x'].values[mask])
        lim = max(np.percentile(sx, 99), np.percentile(ex, 99)) * 1.1
        ax6.scatter(sx, ex, s=3, alpha=0.3, color='#673AB7')
        ax6.plot([0, lim], [0, lim],   'k--', lw=0.8, label='1σ')
        ax6.plot([0, lim], [0, 3*lim], 'r:',  lw=0.8, label='3σ')
        ax6.set_xlim(0, lim); ax6.set_ylim(0, lim)
    ax6.set_xlabel('ESKF σ_x (m)'); ax6.set_ylabel('|err_x| (m)')
    ax6.set_title('Covariance Consistency (X)'); ax6.legend(fontsize=9)

    # ── Global Summary Text ───────────────────────────────────────────────────
    if mask.any():
        h_run = df['err_h_norm'].values[mask]
        summary = (
            f'RUNNING summary — Samples: {mask.sum()}'
            f'   H-RMS: {np.sqrt(np.mean(h_run**2)):.2f} m'
            f'   H-Mean: {np.mean(h_run):.2f} m'
            f'   H-Max: {np.max(h_run):.2f} m'
            f'   H-P95: {np.percentile(h_run, 95):.2f} m'
        )
        if 'tercom_accepted_count' in df.columns:
            summary += f'   TERCOM Fixes: {int(df["tercom_accepted_count"].iloc[-1])}'
        fig.suptitle(f'TERCOM Navigation — Performance Summary\n{summary}',
                     fontsize=13, y=1.01)
    else:
        fig.suptitle('TERCOM Navigation — Performance Summary', fontsize=13)

    save_fig(fig, outdir, '15_summary_dashboard')


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(df: pd.DataFrame) -> dict:
    """Compute all key performance metrics from the dataframe. Returns a dict."""
    from datetime import datetime
    s = {}
    s['n_total']     = len(df)
    s['duration_s']  = float(df['time_s'].iloc[-1] - df['time_s'].iloc[0])
    s['generated']   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    mask = running_mask(df)
    s['n_running'] = int(mask.sum())
    s['running_frac'] = s['n_running'] / max(s['n_total'], 1)

    # Frame alignment offset between MAVROS (PX4 boot frame) and ESKF ENU frame.
    # Reconstructed as the constant difference: true_x_aligned - true_x.
    # A large offset here is EXPECTED and normal — it does NOT indicate error.
    # All error columns in the CSV already have this offset applied correctly.
    if all(c in df.columns for c in ['true_x_aligned', 'true_x',
                                      'true_y_aligned', 'true_y',
                                      'true_z_aligned', 'true_z']):
        s['frame_offset_x'] = float((df['true_x_aligned'] - df['true_x']).mean())
        s['frame_offset_y'] = float((df['true_y_aligned'] - df['true_y']).mean())
        s['frame_offset_z'] = float((df['true_z_aligned'] - df['true_z']).mean())
        s['frame_offset_norm'] = float(np.hypot(s['frame_offset_x'], s['frame_offset_y']))
    else:
        s['frame_offset_x'] = s['frame_offset_y'] = s['frame_offset_z'] = float('nan')
        s['frame_offset_norm'] = float('nan')

    # State durations
    segs = state_segments(df)
    s['state_durations'] = {}
    for t0, t1, st in segs:
        s['state_durations'][st] = s['state_durations'].get(st, 0.0) + (t1 - t0)

    # Horizontal error (RUNNING phase)
    if mask.any():
        h = df['err_h_norm'].values[mask]
        s['h_rms']  = float(np.sqrt(np.mean(h**2)))
        s['h_mean'] = float(np.mean(h))
        s['h_max']  = float(np.max(h))
        s['h_p50']  = float(np.percentile(h, 50))
        s['h_p95']  = float(np.percentile(h, 95))
        s['h_p99']  = float(np.percentile(h, 99))
    else:
        for k in ('h_rms', 'h_mean', 'h_max', 'h_p50', 'h_p95', 'h_p99'):
            s[k] = float('nan')

    # Vertical error (RUNNING phase)
    if mask.any() and 'err_v_abs' in df.columns:
        v = df['err_v_abs'].values[mask]
        s['v_rms']  = float(np.sqrt(np.mean(v**2)))
        s['v_mean'] = float(np.mean(v))
        s['v_max']  = float(np.max(v))
        s['v_p95']  = float(np.percentile(v, 95))
    else:
        for k in ('v_rms', 'v_mean', 'v_max', 'v_p95'):
            s[k] = float('nan')

    # NIS consistency
    if 'nis' in df.columns and mask.any():
        nis = df['nis'].values[mask]
        valid = nis > 0
        if valid.sum() > 5:
            nis_v = nis[valid]
            s['nis_mean'] = float(np.mean(nis_v))
            s['nis_median'] = float(np.median(nis_v))
            s['nis_consistent_frac'] = float(np.mean(
                (nis_v >= CHI2_DOF2_05) & (nis_v <= CHI2_DOF2_95)))
        else:
            s['nis_mean'] = s['nis_median'] = s['nis_consistent_frac'] = float('nan')
    else:
        s['nis_mean'] = s['nis_median'] = s['nis_consistent_frac'] = float('nan')

    # TERCOM acceptance stats
    if 'tercom_accepted_count' in df.columns:
        s['tercom_total_accepted'] = int(df['tercom_accepted_count'].iloc[-1])
        s['tercom_fix_rate_hz'] = s['tercom_total_accepted'] / max(s['duration_s'], 1)
    else:
        s['tercom_total_accepted'] = 0
        s['tercom_fix_rate_hz'] = float('nan')

    # TERCOM quality (RUNNING, accepted only — where MAD > 0)
    if 'tercom_mad' in df.columns and mask.any():
        mad_run = df['tercom_mad'].values[mask]
        active  = mad_run > 0
        if active.sum() > 0:
            s['mad_mean']  = float(np.mean(mad_run[active]))
            s['mad_median']= float(np.median(mad_run[active]))
            s['mad_max']   = float(np.max(mad_run[active]))
        else:
            s['mad_mean'] = s['mad_median'] = s['mad_max'] = float('nan')
    else:
        s['mad_mean'] = s['mad_median'] = s['mad_max'] = float('nan')

    # Covariance consistency: fraction of time |err| < 3σ (per axis, RUNNING)
    if mask.any():
        for ax_e, ax_c in [('err_x', 'cov_xx'), ('err_y', 'cov_yy'), ('err_z', 'cov_zz')]:
            if ax_c in df.columns:
                sig3 = 3.0 * np.sqrt(np.abs(df[ax_c].values[mask]))
                err  = np.abs(df[ax_e].values[mask])
                s[f'within3sigma_{ax_e[-1]}'] = float(np.mean(err < sig3))
            else:
                s[f'within3sigma_{ax_e[-1]}'] = float('nan')
    else:
        for ax in ('x', 'y', 'z'):
            s[f'within3sigma_{ax}'] = float('nan')

    # Speed
    if 'true_speed' in df.columns:
        s['mean_speed'] = float(df['true_speed'].mean())
        s['max_speed']  = float(df['true_speed'].max())
    else:
        s['mean_speed'] = s['max_speed'] = float('nan')

    return s


# ─────────────────────────────────────────────────────────────────────────────
# PDF report generator
# ─────────────────────────────────────────────────────────────────────────────

def _page_style(fig):
    """Apply a consistent dark-header style to a report page figure."""
    fig.patch.set_facecolor('white')


def _header_bar(fig, title: str, subtitle: str = ''):
    """Draw a colored header bar at the top of a report page."""
    ax = fig.add_axes([0, 0.93, 1, 0.07])
    ax.set_facecolor('#1A237E')
    ax.axis('off')
    ax.text(0.015, 0.55, title, transform=ax.transAxes,
            color='white', fontsize=14, fontweight='bold', va='center')
    if subtitle:
        ax.text(0.015, 0.10, subtitle, transform=ax.transAxes,
                color='#90CAF9', fontsize=9, va='center')


def _section_label(ax, text: str):
    ax.set_title(text, fontsize=11, fontweight='bold', loc='left', pad=6,
                 color='#1A237E')


def _fmt(val, unit='m', decimals=2):
    if np.isnan(val):
        return '—'
    return f'{val:.{decimals}f} {unit}'.strip()


def _verdict(condition: bool, good: str, bad: str) -> tuple:
    """Return (text, color) for a pass/fail verdict."""
    return (good, '#2E7D32') if condition else (bad, '#C62828')


def generate_pdf_report(df: pd.DataFrame, stats: dict,
                        csv_path: str, outdir: str,
                        origin_lat: float = None,
                        origin_lon: float = None,
                        satellite_zoom: int = 17) -> str:
    """Generate a multi-page PDF analysis report.

    Returns the path to the saved PDF.
    """
    out_path = os.path.join(outdir, 'tercom_analysis_report.pdf')
    mask = running_mask(df)
    t    = df['time_s'].values
    segs = state_segments(df)

    with PdfPages(out_path) as pdf:

        # ── PAGE 1: Title page ──────────────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        _page_style(fig)

        # Navy banner
        banner = fig.add_axes([0, 0.72, 1, 0.28])
        banner.set_facecolor('#0D1B5E')
        banner.axis('off')
        banner.text(0.5, 0.65, 'TERCOM GPS-Denied Navigation System',
                    ha='center', va='center', color='white',
                    fontsize=20, fontweight='bold', transform=banner.transAxes)
        banner.text(0.5, 0.35, 'Performance Analysis Report',
                    ha='center', va='center', color='#90CAF9',
                    fontsize=15, transform=banner.transAxes)

        body = fig.add_axes([0.08, 0.28, 0.84, 0.40])
        body.axis('off')

        meta = [
            ('Log file',          os.path.basename(csv_path)),
            ('Report generated',  stats['generated']),
            ('Total samples',     f"{stats['n_total']:,}"),
            ('Flight duration',   f"{stats['duration_s']:.1f} s  "
                                  f"({stats['duration_s']/60:.1f} min)"),
            ('RUNNING samples',   f"{stats['n_running']:,}  "
                                  f"({stats['running_frac']*100:.1f}% of flight)"),
            ('TERCOM fixes',      f"{stats['tercom_total_accepted']}  "
                                  f"({stats['tercom_fix_rate_hz']:.3f} Hz)"),
        ]
        for i, (k, v) in enumerate(meta):
            y = 0.88 - i * 0.12
            body.text(0.02, y, k + ':', fontsize=11, fontweight='bold', color='#333')
            body.text(0.35, y, v,        fontsize=11, color='#111')

        # Key result box
        kpi = fig.add_axes([0.08, 0.08, 0.84, 0.18])
        kpi.set_facecolor('#E8EAF6')
        kpi.set_xlim(0, 1); kpi.set_ylim(0, 1); kpi.axis('off')
        kpi.add_patch(mpatches.FancyBboxPatch(
            (0.01, 0.05), 0.98, 0.9, boxstyle='round,pad=0.02',
            fc='#E8EAF6', ec='#3949AB', lw=1.5))
        kpis = [
            ('H-RMS',    _fmt(stats['h_rms'])),
            ('H-P95',    _fmt(stats['h_p95'])),
            ('H-Max',    _fmt(stats['h_max'])),
            ('V-RMS',    _fmt(stats['v_rms'])),
            ('NIS cons.',f"{stats['nis_consistent_frac']*100:.1f}%" if not np.isnan(stats['nis_consistent_frac']) else '—'),
            ('Avg Speed',_fmt(stats['mean_speed'], 'm/s')),
        ]
        for j, (label, val) in enumerate(kpis):
            x = 0.05 + j * 0.16
            kpi.text(x + 0.05, 0.72, label, ha='center', fontsize=9,
                     color='#3949AB', fontweight='bold')
            kpi.text(x + 0.05, 0.30, val,   ha='center', fontsize=13,
                     color='#1A237E', fontweight='bold')

        fig.text(0.5, 0.03, 'TERCOM Navigation Analysis  •  tercom_nav ROS 2 Package',
                 ha='center', fontsize=8, color='#888')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ── PAGE 2: Executive Summary ───────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        _page_style(fig)
        _header_bar(fig, 'Executive Summary',
                    'Key performance indicators — RUNNING phase only unless noted')

        ax = fig.add_axes([0.04, 0.06, 0.92, 0.84])
        ax.axis('off')

        # Build table rows
        rows = [
            # ── Position Accuracy ──────────────────────────────────────
            ['POSITION ACCURACY (RUNNING phase)', '', '', ''],
            ['Horizontal RMS error',
             _fmt(stats['h_rms']), 'Target < 50 m',
             '✓ PASS' if stats['h_rms'] < 50 else '✗ FAIL'],
            ['Horizontal Mean error',
             _fmt(stats['h_mean']), '', ''],
            ['Horizontal 95th percentile',
             _fmt(stats['h_p95']), '', ''],
            ['Horizontal 99th percentile',
             _fmt(stats['h_p99']), '', ''],
            ['Horizontal Max error',
             _fmt(stats['h_max']), '', ''],
            ['Vertical RMS error',
             _fmt(stats['v_rms']), 'Target < 20 m', ''],
            ['Vertical 95th percentile',
             _fmt(stats['v_p95']), '', ''],
            # ── Filter Consistency ─────────────────────────────────────
            ['FILTER CONSISTENCY (NIS)', '', '', ''],
            ['NIS mean',
             f"{stats['nis_mean']:.3f}" if not np.isnan(stats['nis_mean']) else '—',
             f'χ²(2) bounds [{CHI2_DOF2_05:.2f}, {CHI2_DOF2_95:.2f}]', ''],
            ['NIS median',
             f"{stats['nis_median']:.3f}" if not np.isnan(stats['nis_median']) else '—',
             '', ''],
            ['Samples within χ² bounds',
             f"{stats['nis_consistent_frac']*100:.1f}%" if not np.isnan(stats['nis_consistent_frac']) else '—',
             'Target > 80%',
             '✓ Consistent' if not np.isnan(stats['nis_consistent_frac']) and stats['nis_consistent_frac'] > 0.80 else '⚠ Inconsistent'],
            # ── Covariance Bounds ──────────────────────────────────────
            ['COVARIANCE VALIDITY (% time |err| < 3σ)', '', '', ''],
            ['X axis (East)',
             f"{stats['within3sigma_x']*100:.1f}%" if not np.isnan(stats['within3sigma_x']) else '—',
             'Target > 99%',
             '✓' if not np.isnan(stats['within3sigma_x']) and stats['within3sigma_x'] > 0.99 else '⚠'],
            ['Y axis (North)',
             f"{stats['within3sigma_y']*100:.1f}%" if not np.isnan(stats['within3sigma_y']) else '—',
             'Target > 99%',
             '✓' if not np.isnan(stats['within3sigma_y']) and stats['within3sigma_y'] > 0.99 else '⚠'],
            ['Z axis (Up)',
             f"{stats['within3sigma_z']*100:.1f}%" if not np.isnan(stats['within3sigma_z']) else '—',
             'Target > 99%',
             '✓' if not np.isnan(stats['within3sigma_z']) and stats['within3sigma_z'] > 0.99 else '⚠'],
            # ── TERCOM Match Quality ───────────────────────────────────
            ['TERCOM MATCH QUALITY', '', '', ''],
            ['Total accepted fixes',
             str(stats['tercom_total_accepted']), '', ''],
            ['Average acceptance rate',
             f"{stats['tercom_fix_rate_hz']:.3f} Hz" if not np.isnan(stats['tercom_fix_rate_hz']) else '—',
             '', ''],
            ['Mean MAD (accepted fixes)',
             _fmt(stats['mad_mean']), 'Reject threshold: 30 m', ''],
            ['Max MAD seen',
             _fmt(stats['mad_max']), '', ''],
            # ── Frame alignment offset ─────────────────────────────────
            ['FRAME ALIGNMENT (MAVROS\u2192ENU)', '', '', ''],
            ['Horizontal offset \u2016\u0394E,\u0394N\u2016',
             _fmt(stats.get('frame_offset_norm', float('nan'))),
             'Constant PX4-boot\u2192UTM shift; removed from all error columns', ''],
            ['\u0394East offset',
             _fmt(stats.get('frame_offset_x', float('nan'))), '', ''],
            ['\u0394North offset',
             _fmt(stats.get('frame_offset_y', float('nan'))), '', ''],
            ['\u0394Up offset',
             _fmt(stats.get('frame_offset_z', float('nan'))), '', ''],
            # ── State durations ────────────────────────────────────────
            ['FILTER STATE DURATIONS', '', '', ''],
        ]
        for st, dur in sorted(stats['state_durations'].items(), key=lambda x: -x[1]):
            rows.append([f'  {st}',
                         f'{dur:.1f} s',
                         f'{dur/stats["duration_s"]*100:.1f}%', ''])

        col_widths   = [0.44, 0.17, 0.25, 0.14]
        col_headers  = ['Metric', 'Value', 'Reference', 'Status']
        col_colors   = ['#1A237E'] * 4

        y = 0.97
        # Header row
        x = 0.0
        for hdr, w, col in zip(col_headers, col_widths, col_colors):
            rect = mpatches.FancyBboxPatch(
                (x + 0.002, y - 0.04), w - 0.004, 0.04,
                boxstyle='square,pad=0', fc='#1A237E', ec='none',
                transform=ax.transAxes, clip_on=False)
            ax.add_patch(rect)
            ax.text(x + w/2, y - 0.015, hdr,
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color='white', transform=ax.transAxes)
            x += w
        y -= 0.045

        # Data rows
        row_h = 0.039
        for ri, row in enumerate(rows):
            is_section = row[1] == '' and row[2] == '' and row[3] == ''
            bg = '#C5CAE9' if is_section else ('#F5F5F5' if ri % 2 == 0 else 'white')
            x = 0.0
            for ci, (cell, w) in enumerate(zip(row, col_widths)):
                rect = mpatches.FancyBboxPatch(
                    (x + 0.001, y - row_h + 0.003), w - 0.002, row_h - 0.004,
                    boxstyle='square,pad=0', fc=bg, ec='#E0E0E0', lw=0.3,
                    transform=ax.transAxes, clip_on=False)
                ax.add_patch(rect)
                fw = 'bold' if is_section or ci == 0 else 'normal'
                fc = '#1A237E' if is_section else (
                    '#2E7D32' if '✓' in str(cell) else
                    '#C62828' if '✗' in str(cell) else
                    '#E65100' if '⚠' in str(cell) else '#222')
                ha = 'center' if ci > 0 else 'left'
                xtext = x + (0.01 if ha == 'left' else w/2)
                ax.text(xtext, y - row_h/2, str(cell),
                        ha=ha, va='center', fontsize=8,
                        fontweight=fw, color=fc, transform=ax.transAxes)
                x += w
            y -= row_h
            if y < 0.02:
                break  # guard against overflowing the page

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ── PAGE 3: Trajectory Analysis ─────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        _page_style(fig)
        _header_bar(fig, 'Trajectory Analysis', 'XY ground track and 3D flight path')

        # Use frame-aligned ground truth for all trajectory plots
        _tx = df['true_x_aligned'] if 'true_x_aligned' in df.columns else df['true_x']
        _ty = df['true_y_aligned'] if 'true_y_aligned' in df.columns else df['true_y']
        _tz = df['true_z_aligned'] if 'true_z_aligned' in df.columns else df['true_z']

        ax_xy = fig.add_axes([0.05, 0.10, 0.42, 0.78])
        ax_xy.plot(_tx, _ty, color='#2196F3',
                   lw=1.6, label='Ground Truth (aligned)')
        ax_xy.plot(df['est_x'],  df['est_y'],  color='#FF5722',
                   lw=1.2, ls='--', label='ESKF Estimate')
        ax_xy.scatter(_tx.iloc[0], _ty.iloc[0],
                      s=80, color='green', zorder=5, label='Start')
        ax_xy.scatter(_tx.iloc[-1], _ty.iloc[-1],
                      s=80, color='red',   zorder=5, marker='s', label='End')
        if 'tercom_accepted_count' in df.columns:
            fix_rows = df[df['tercom_accepted_count'].diff().fillna(0) > 0]
            ax_xy.scatter(fix_rows['est_x'], fix_rows['est_y'],
                          marker='+', s=60, color='lime', lw=1.5,
                          zorder=4, label='TERCOM Fix')
        # Annotate the frame offset
        if not np.isnan(stats.get('frame_offset_norm', float('nan'))):
            off_txt = (f"Frame offset (MAVROS→ENU):\n"
                       f"ΔE={stats['frame_offset_x']:.1f} m, "
                       f"ΔN={stats['frame_offset_y']:.1f} m\n"
                       f"‖offset‖={stats['frame_offset_norm']:.1f} m")
            ax_xy.text(0.02, 0.02, off_txt, transform=ax_xy.transAxes, fontsize=7,
                       va='bottom', color='#555',
                       bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        ax_xy.set_xlabel('East (m)'); ax_xy.set_ylabel('North (m)')
        ax_xy.set_aspect('equal', adjustable='datalim')
        ax_xy.legend(fontsize=8, loc='best')
        _section_label(ax_xy, 'XY Ground Track (frame-aligned)')

        ax3d = fig.add_axes([0.54, 0.10, 0.44, 0.78], projection='3d')
        ax3d.plot(_tx, _ty, _tz,
                  color='#2196F3', lw=1.2, label='Ground Truth (aligned)')
        ax3d.plot(df['est_x'],  df['est_y'],  df['est_z'],
                  color='#FF5722', lw=0.9, ls='--', label='ESKF Estimate')
        ax3d.set_xlabel('E (m)', fontsize=8); ax3d.set_ylabel('N (m)', fontsize=8)
        ax3d.set_zlabel('U (m)', fontsize=8)
        ax3d.tick_params(labelsize=7)
        ax3d.legend(fontsize=8)
        ax3d.set_title('3D Trajectory', fontsize=11, fontweight='bold',
                       color='#1A237E', pad=4)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ── PAGE 3b: Satellite Map Trajectory ───────────────────────────────
        if origin_lat is not None and origin_lon is not None:
            try:
                import contextily as ctx
                from pyproj import Transformer

                utm_zone = int((origin_lon + 180.0) / 6.0) + 1
                utm_epsg = 32600 + utm_zone if origin_lat >= 0 else 32700 + utm_zone
                tf_geo_to_utm = Transformer.from_crs('EPSG:4326', f'EPSG:{utm_epsg}',
                                                      always_xy=True)
                tf_utm_to_web = Transformer.from_crs(f'EPSG:{utm_epsg}', 'EPSG:3857',
                                                      always_xy=True)
                origin_e, origin_n = tf_geo_to_utm.transform(origin_lon, origin_lat)

                def _enu_to_web(x_enu, y_enu):
                    return tf_utm_to_web.transform(origin_e + x_enu, origin_n + y_enu)

                _tx_s = df['true_x_aligned'] if 'true_x_aligned' in df.columns else df['true_x']
                _ty_s = df['true_y_aligned'] if 'true_y_aligned' in df.columns else df['true_y']
                gt_wx, gt_wy   = _enu_to_web(_tx_s.values, _ty_s.values)
                est_wx, est_wy = _enu_to_web(df['est_x'].values, df['est_y'].values)

                fig = plt.figure(figsize=(11, 8.5))
                _page_style(fig)
                _header_bar(fig, 'Satellite Map Trajectory',
                            f'Origin: {origin_lat:.6f}°N, {origin_lon:.6f}°E  '
                            f'(UTM zone {utm_zone}{"N" if origin_lat >= 0 else "S"})')

                ax_sat = fig.add_axes([0.05, 0.05, 0.90, 0.83])
                ax_sat.plot(gt_wx, gt_wy, color='#2196F3', lw=2.0,
                            label='Ground Truth (frame-aligned)', zorder=3)
                ax_sat.plot(est_wx, est_wy, color='#FF5722', lw=1.4, ls='--',
                            label='ESKF Estimate', zorder=4)
                ax_sat.scatter(gt_wx[0], gt_wy[0], marker='o', s=90,
                               color='lime', edgecolors='k', lw=0.8, zorder=6,
                               label='Start')
                ax_sat.scatter(gt_wx[-1], gt_wy[-1], marker='s', s=90,
                               color='red', edgecolors='k', lw=0.8, zorder=6,
                               label='End')
                if 'tercom_accepted_count' in df.columns:
                    fix_rows = df[df['tercom_accepted_count'].diff().fillna(0) > 0]
                    if not fix_rows.empty:
                        fx, fy = _enu_to_web(fix_rows['est_x'].values,
                                             fix_rows['est_y'].values)
                        ax_sat.scatter(fx, fy, marker='+', s=60, color='yellow',
                                       linewidths=1.8, zorder=5,
                                       label='TERCOM Fix Applied')
                try:
                    ctx.add_basemap(ax_sat, crs='EPSG:3857',
                                    source=ctx.providers.Esri.WorldImagery,
                                    zoom=satellite_zoom, attribution_size=6)
                except Exception as _te:
                    ax_sat.text(0.5, 0.5, f'Satellite tiles unavailable:\n{_te}',
                                transform=ax_sat.transAxes, ha='center', va='center',
                                color='red', fontsize=9,
                                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))
                ax_sat.set_xlabel('Web Mercator X (m)', fontsize=9)
                ax_sat.set_ylabel('Web Mercator Y (m)', fontsize=9)
                ax_sat.legend(fontsize=8, loc='best', framealpha=0.85)
                _section_label(ax_sat, 'XY Ground Track — Satellite Basemap')

                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            except ImportError:
                pass  # contextily/pyproj not available; skip page silently

        # ── PAGE 4: Position Error Analysis ─────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        _page_style(fig)
        _header_bar(fig, 'Position Error Analysis',
                    'Instantaneous errors vs time and statistical distributions')

        gs = GridSpec(2, 2, figure=fig, top=0.88, bottom=0.07,
                      left=0.07, right=0.97, hspace=0.40, wspace=0.30)

        ax1 = fig.add_subplot(gs[0, :])
        add_state_background(ax1, segs)
        ax1.plot(t, df['err_h_norm'], color='#E91E63', lw=1.1, label='Horizontal ‖Δh‖')
        if 'err_v_abs' in df.columns:
            ax1.plot(t, df['err_v_abs'], color='#9C27B0', lw=1.0,
                     ls='--', label='Vertical |Δz|')
        ax1.plot(t, df['err_x'], color='#F44336', lw=0.7, alpha=0.6, label='ΔX')
        ax1.plot(t, df['err_y'], color='#4CAF50', lw=0.7, alpha=0.6, label='ΔY')
        ax1.axhline(0, color='k', lw=0.5, ls=':')
        ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Error (m)')
        ax1.legend(ncol=5, fontsize=8, loc='upper right')
        _section_label(ax1, 'Position Error vs Time')

        ax2 = fig.add_subplot(gs[1, 0])
        if mask.any():
            h_err = df['err_h_norm'].values[mask]
            ax2.hist(h_err, bins=50, color='#E91E63', alpha=0.75,
                     edgecolor='white', lw=0.3)
            ax2.axvline(stats['h_rms'],  color='k',    lw=1.4, ls='--',
                        label=f"RMS={stats['h_rms']:.1f}m")
            ax2.axvline(stats['h_p50'],  color='navy', lw=1.2, ls=':',
                        label=f"P50={stats['h_p50']:.1f}m")
            ax2.axvline(stats['h_p95'],  color='orange', lw=1.2, ls='-.',
                        label=f"P95={stats['h_p95']:.1f}m")
            ax2.set_xlabel('Horizontal Error (m)'); ax2.set_ylabel('Count')
            ax2.legend(fontsize=8)
        _section_label(ax2, 'H-Error Distribution (RUNNING)')

        ax3 = fig.add_subplot(gs[1, 1])
        if mask.any() and 'err_v_abs' in df.columns:
            v_err = df['err_v_abs'].values[mask]
            ax3.hist(v_err, bins=50, color='#9C27B0', alpha=0.75,
                     edgecolor='white', lw=0.3)
            ax3.axvline(stats['v_rms'],  color='k',    lw=1.4, ls='--',
                        label=f"RMS={stats['v_rms']:.1f}m")
            ax3.axvline(stats['v_p95'],  color='orange', lw=1.2, ls='-.',
                        label=f"P95={stats['v_p95']:.1f}m")
            ax3.set_xlabel('Vertical Error (m)'); ax3.set_ylabel('Count')
            ax3.legend(fontsize=8)
        _section_label(ax3, 'V-Error Distribution (RUNNING)')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ── PAGE 5: Filter Consistency ───────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        _page_style(fig)
        _header_bar(fig, 'Filter Consistency Analysis',
                    'NIS chi-squared test and covariance vs actual error bounds')

        gs = GridSpec(2, 3, figure=fig, top=0.88, bottom=0.07,
                      left=0.07, right=0.97, hspace=0.42, wspace=0.30)

        # NIS plot (full width top)
        ax_nis = fig.add_subplot(gs[0, :])
        if 'nis' in df.columns:
            add_state_background(ax_nis, segs)
            ax_nis.plot(t, df['nis'], color='#673AB7', lw=1.0, label='NIS')
            ax_nis.axhline(CHI2_DOF2_95, color='#F44336', lw=1.2, ls='--',
                           label=f'χ²(2, 0.95) = {CHI2_DOF2_95:.2f}')
            ax_nis.axhline(CHI2_DOF2_05, color='#4CAF50', lw=1.2, ls='--',
                           label=f'χ²(2, 0.05) = {CHI2_DOF2_05:.2f}')
            ax_nis.fill_between(t, CHI2_DOF2_05, CHI2_DOF2_95,
                                alpha=0.10, color='#4CAF50')
            ax_nis.set_ylim(bottom=0)
        ax_nis.set_xlabel('Time (s)'); ax_nis.set_ylabel('NIS')
        ax_nis.legend(ncol=4, fontsize=8)
        _section_label(ax_nis, f"NIS — Consistency: "
                       f"{stats['nis_consistent_frac']*100:.1f}% within bounds"
                       if not np.isnan(stats['nis_consistent_frac']) else 'NIS')

        # Covariance consistency per axis
        for ci, (err_col, cov_col, lbl) in enumerate([
            ('err_x', 'cov_xx', 'X (East)'),
            ('err_y', 'cov_yy', 'Y (North)'),
            ('err_z', 'cov_zz', 'Z (Up)'),
        ]):
            ax_c = fig.add_subplot(gs[1, ci])
            add_state_background(ax_c, segs)
            if cov_col in df.columns:
                sigma = np.sqrt(np.abs(df[cov_col].values))
                ax_c.fill_between(t, -3*sigma, 3*sigma,
                                  alpha=0.18, color='#2196F3', label='±3σ')
                ax_c.plot(t, df[err_col].values, color='#F44336', lw=0.8,
                          label=f'err_{lbl[0].lower()}')
                ax_c.axhline(0, color='k', lw=0.4, ls=':')
            ax_c.set_xlabel('Time (s)'); ax_c.set_ylabel('Error (m)')
            ax_c.legend(fontsize=7)
            within = stats.get(f"within3sigma_{err_col[-1]}", float('nan'))
            _section_label(ax_c, f'{lbl} — {within*100:.0f}% within 3σ'
                           if not np.isnan(within) else lbl)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ── PAGE 6: TERCOM Quality Analysis ─────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        _page_style(fig)
        _header_bar(fig, 'TERCOM Match Quality Analysis',
                    'Match metrics over time and their correlation with position error')

        gs = GridSpec(3, 2, figure=fig, top=0.88, bottom=0.07,
                      left=0.07, right=0.97, hspace=0.45, wspace=0.30)

        quality_cols = [
            ('tercom_mad',       'MAD (m)',          '#FF9800', 30.0),
            ('tercom_disc',      'Discrimination',   '#2196F3', 1.02),
            ('tercom_roughness', 'Roughness (m)',     '#4CAF50', 5.0),
            ('tercom_noise',     'Noise (m)',         '#9C27B0', None),
        ]
        for ri, (col, ylabel, color, thresh) in enumerate(quality_cols):
            r, c = divmod(ri, 2)
            ax_q = fig.add_subplot(gs[r, c])
            if col in df.columns:
                add_state_background(ax_q, segs)
                ax_q.plot(t, df[col], color=color, lw=0.9)
                if thresh is not None:
                    ax_q.axhline(thresh, color='k', lw=0.9, ls='--',
                                 label=f'threshold={thresh}')
                    ax_q.legend(fontsize=8)
            ax_q.set_xlabel('Time (s)'); ax_q.set_ylabel(ylabel)
            _section_label(ax_q, ylabel + ' vs Time')

        # MAD vs Error scatter
        ax_sc = fig.add_subplot(gs[2, :])
        if 'tercom_mad' in df.columns and mask.any():
            mad_v  = df['tercom_mad'].values[mask]
            herr_v = df['err_h_norm'].values[mask]
            active = mad_v > 0
            if active.sum() > 5:
                ax_sc.scatter(mad_v[active], herr_v[active],
                              s=5, alpha=0.35, color='#FF9800', label='Samples')
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    cf = np.polyfit(mad_v[active], herr_v[active], 1)
                xs = np.linspace(mad_v[active].min(), mad_v[active].max(), 200)
                ax_sc.plot(xs, np.poly1d(cf)(xs), 'k--', lw=1.3,
                           label=f'Fit: err = {cf[0]:.2f}·MAD + {cf[1]:.2f}')
                ax_sc.axvline(30.0, color='r', lw=1, ls=':', label='MAD reject (30 m)')
                ax_sc.set_xlabel('TERCOM MAD (m)')
                ax_sc.set_ylabel('Horizontal Error (m)')
                ax_sc.legend(fontsize=8)
        _section_label(ax_sc, 'TERCOM MAD vs Position Error (RUNNING phase)')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ── PAGE 7: System Health & Speed ────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        _page_style(fig)
        _header_bar(fig, 'System Health & Flight Profile',
                    'Filter health metrics, state timeline, and vehicle speed')

        gs = GridSpec(4, 1, figure=fig, top=0.88, bottom=0.07,
                      left=0.07, right=0.97, hspace=0.55)

        # State timeline
        ax_st = fig.add_subplot(gs[0])
        for t0, t1, st in segs:
            ax_st.barh(0, t1 - t0, left=t0, height=0.8,
                       color=STATE_COLORS.get(st, '#ccc'))
            if (t1 - t0) > stats['duration_s'] * 0.04:
                ax_st.text((t0 + t1)/2, 0, st, ha='center', va='center',
                           fontsize=7, fontweight='bold',
                           color='white' if st in ('RUNNING', 'DIVERGED') else 'black')
        ax_st.set_xlim(t[0], t[-1])
        ax_st.set_yticks([])
        ax_st.set_xlabel('Time (s)')
        _section_label(ax_st, 'Filter State Timeline')

        # Health metrics
        health_cols = [
            ('health_max_pos_std', 'Max Pos σ (m)', '#E91E63'),
            ('health_innov_norm',  'Innovation Norm', '#FF9800'),
            ('health_is_healthy',  'Healthy (1=Yes)', '#4CAF50'),
        ]
        for ri, (col, ylabel, color) in enumerate(health_cols):
            ax_h = fig.add_subplot(gs[ri + 1])
            if col in df.columns:
                add_state_background(ax_h, segs)
                ax_h.plot(t, df[col], color=color, lw=0.9)
            ax_h.set_xlabel('Time (s)'); ax_h.set_ylabel(ylabel, fontsize=9)
            _section_label(ax_h, ylabel)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ── PAGE 8: Error Statistics & Covariance Scatter ───────────────────
        fig = plt.figure(figsize=(11, 8.5))
        _page_style(fig)
        _header_bar(fig, 'Error Statistics & Covariance Scatter',
                    'Sliding-window statistics and per-axis covariance validation')

        gs = GridSpec(2, 3, figure=fig, top=0.88, bottom=0.07,
                      left=0.07, right=0.97, hspace=0.42, wspace=0.30)

        # Sliding-window stats (top row, full width)
        ax_stat = fig.add_subplot(gs[0, :])
        h_all = df['err_h_norm'].values
        dt_avg = (t[-1] - t[0]) / max(len(t) - 1, 1)
        win = max(10, int(5.0 / dt_avg))
        rms_sw  = np.array([np.sqrt(np.mean(h_all[max(0, i-win):i+1]**2))
                            for i in range(len(h_all))])
        mean_sw = np.array([np.mean(h_all[max(0, i-win):i+1])
                            for i in range(len(h_all))])
        max_sw  = np.array([np.max(h_all[max(0, i-win):i+1])
                            for i in range(len(h_all))])
        add_state_background(ax_stat, segs)
        ax_stat.fill_between(t, 0, max_sw, alpha=0.12, color='#F44336')
        ax_stat.plot(t, max_sw,  color='#F44336', lw=0.8, ls=':', label='Sliding Max')
        ax_stat.plot(t, rms_sw,  color='#FF9800', lw=1.4, label=f'Sliding RMS ({win}-sample)')
        ax_stat.plot(t, mean_sw, color='#4CAF50', lw=1.4, label='Sliding Mean')
        ax_stat.plot(t, h_all,   color='#9E9E9E', lw=0.5, alpha=0.5, label='Instantaneous')
        ax_stat.set_xlabel('Time (s)'); ax_stat.set_ylabel('Horizontal Error (m)')
        ax_stat.legend(ncol=4, fontsize=8)
        _section_label(ax_stat, 'Sliding-Window Error Statistics')

        # Per-axis covariance scatter (bottom row)
        for ci, (err_col, cov_col, lbl) in enumerate([
            ('err_x', 'cov_xx', 'X East'),
            ('err_y', 'cov_yy', 'Y North'),
            ('err_z', 'cov_zz', 'Z Up'),
        ]):
            ax_cv = fig.add_subplot(gs[1, ci])
            if mask.any() and cov_col in df.columns:
                sx  = np.sqrt(np.abs(df[cov_col].values[mask]))
                ex  = np.abs(df[err_col].values[mask])
                lim = max(np.percentile(sx, 99), np.percentile(ex, 99)) * 1.05
                ax_cv.scatter(sx, ex, s=3, alpha=0.25, color='#673AB7')
                ax_cv.plot([0, lim], [0, lim],   'k--', lw=0.8, label='1σ')
                ax_cv.plot([0, lim], [0, 2*lim], 'b:',  lw=0.8, label='2σ')
                ax_cv.plot([0, lim], [0, 3*lim], 'r-',  lw=0.8, label='3σ')
                ax_cv.set_xlim(0, lim); ax_cv.set_ylim(0, lim)
                ax_cv.set_xlabel(f'ESKF σ_{lbl[0].lower()} (m)')
                ax_cv.set_ylabel(f'|err_{lbl[0].lower()}| (m)')
                ax_cv.legend(fontsize=7)
            _section_label(ax_cv, f'Cov Scatter — {lbl}')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ── PAGE 9: Conclusions & Algorithmic Analysis ───────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        _page_style(fig)
        _header_bar(fig, 'Conclusions & Algorithmic Analysis',
                    'Automated interpretation of results and recommendations')

        ax = fig.add_axes([0.05, 0.04, 0.90, 0.86])
        ax.axis('off')

        # Build conclusion paragraphs dynamically from stats
        conclusions = _build_conclusions(stats)
        y = 0.97
        for section_title, paragraphs in conclusions:
            # Section heading
            ax.text(0.0, y, section_title, fontsize=12, fontweight='bold',
                    color='#1A237E', transform=ax.transAxes, va='top')
            y -= 0.04
            ax.axhline(y + 0.005, color='#1A237E', lw=0.8,
                       xmin=0.0, xmax=1.0)
            y -= 0.005
            for para in paragraphs:
                # Word-wrap manually into ~120-char lines
                words = para.split()
                line, lines = '', []
                for w in words:
                    if len(line) + len(w) + 1 <= 115:
                        line = (line + ' ' + w).lstrip()
                    else:
                        lines.append(line)
                        line = w
                if line:
                    lines.append(line)
                for ln in lines:
                    ax.text(0.015, y, ln, fontsize=9, color='#222',
                            transform=ax.transAxes, va='top')
                    y -= 0.033
                y -= 0.010
            y -= 0.010
            if y < 0.04:
                break

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    print(f'  PDF report: {out_path}')
    return out_path


def _build_conclusions(s: dict) -> list:
    """Return list of (section_title, [paragraph, ...]) for the conclusions page."""
    def pct(v): return f'{v*100:.1f}%' if not np.isnan(v) else 'N/A'
    def fm(v, u='m'): return _fmt(v, u)

    sections = []

    # 1. Overall Accuracy Assessment
    paras = []
    if not np.isnan(s['h_rms']):
        rating = ('excellent (< 20 m)' if s['h_rms'] < 20 else
                  'good (< 50 m)'      if s['h_rms'] < 50 else
                  'marginal (< 100 m)' if s['h_rms'] < 100 else
                  'poor (≥ 100 m)')
        paras.append(
            f"The ESKF achieved a horizontal position RMS error of {fm(s['h_rms'])} "
            f"during the RUNNING phase, classified as {rating}. The 95th percentile "
            f"error was {fm(s['h_p95'])} and the worst-case error reached {fm(s['h_max'])}. "
            f"The vertical (altitude) RMS error was {fm(s['v_rms'])}, with a 95th "
            f"percentile of {fm(s['v_p95'])}."
        )
    if not np.isnan(s['running_frac']):
        paras.append(
            f"The filter operated in RUNNING state for {pct(s['running_frac'])} of "
            f"the total {s['duration_s']:.1f}-second flight. "
            + ('The high RUNNING fraction indicates stable filter convergence.'
               if s['running_frac'] > 0.85
               else 'A low RUNNING fraction may indicate frequent divergence events '
                    'that warrant investigation into the divergence recovery settings.')
        )
    sections.append(('1.  Overall Navigation Accuracy', paras))

    # 2. Filter Consistency (NIS)
    paras = []
    if not np.isnan(s['nis_consistent_frac']):
        consistent = s['nis_consistent_frac'] > 0.80
        paras.append(
            f"The Normalized Innovation Squared (NIS) consistency test yielded "
            f"{pct(s['nis_consistent_frac'])} of RUNNING-phase samples within the "
            f"χ²(2) 5%–95% bounds [{CHI2_DOF2_05:.2f}, {CHI2_DOF2_95:.2f}]. "
            f"A well-tuned filter should have ≥ 80% of samples within bounds. "
            f"This filter is {'consistent — process and measurement noise are well-calibrated' if consistent else 'INCONSISTENT. If NIS is predominantly above the upper bound the filter is over-confident (Q/R too small); if predominantly below it is under-confident (Q/R too large)'}."
        )
        if not consistent:
            nis_m = s['nis_mean']
            if not np.isnan(nis_m):
                if nis_m > CHI2_DOF2_95:
                    paras.append(
                        f"With a mean NIS of {nis_m:.2f} (above upper bound {CHI2_DOF2_95:.2f}), "
                        f"the filter is over-confident. Recommended action: increase the process "
                        f"noise covariance Q or the TERCOM measurement noise R to allow larger "
                        f"innovations without flagging divergence."
                    )
                elif nis_m < CHI2_DOF2_05:
                    paras.append(
                        f"With a mean NIS of {nis_m:.2f} (below lower bound {CHI2_DOF2_05:.2f}), "
                        f"the filter is under-confident (overly conservative). Recommended action: "
                        f"decrease Q or decrease R to make the filter trust its predictions more."
                    )
    sections.append(('2.  Filter Consistency (NIS Chi-Squared Test)', paras))

    # 3. Covariance Validity
    paras = []
    cov_results = []
    for ax_label, key in [('X (East)', 'within3sigma_x'),
                           ('Y (North)', 'within3sigma_y'),
                           ('Z (Up)',    'within3sigma_z')]:
        v = s.get(key, float('nan'))
        if not np.isnan(v):
            status = 'valid' if v > 0.99 else 'inflated' if v > 0.95 else 'INVALID (too optimistic)'
            cov_results.append(f'{ax_label}: {pct(v)} ({status})')
    if cov_results:
        paras.append(
            'Percentage of RUNNING-phase time where the absolute position error remained '
            'within the ESKF 3σ covariance bound (expected > 99.7% for a Gaussian filter): '
            + '; '.join(cov_results) + '. '
            'A value well below 99.7% indicates that the reported covariance is optimistic '
            '(the filter under-reports its uncertainty), which can be dangerous in '
            'safety-critical applications.'
        )
    sections.append(('3.  Covariance Validity', paras))

    # 4. TERCOM Match Quality
    paras = []
    if not np.isnan(s['mad_mean']):
        paras.append(
            f"TERCOM terrain correlation produced {s['tercom_total_accepted']} accepted "
            f"fixes at an average rate of {s['tercom_fix_rate_hz']:.3f} Hz. The mean "
            f"Mean-Absolute-Difference (MAD) for accepted fixes was {fm(s['mad_mean'])} "
            f"(median {fm(s['mad_median'])}), well below the {fm(30.0)} rejection threshold. "
            f"A lower MAD indicates tighter terrain correlation and higher-confidence fixes. "
            f"The maximum MAD observed was {fm(s['mad_max'])}."
        )
        fix_rate = s['tercom_fix_rate_hz']
        if not np.isnan(fix_rate):
            if fix_rate < 0.05:
                paras.append(
                    'The low TERCOM fix rate suggests frequent rejections, possibly due to '
                    'low terrain roughness, discriminaton failures on straight-line segments, '
                    'or a very conservative MAD threshold. Consider lowering roughness_min '
                    'or adjusting the flight path to traverse varied terrain.'
                )
            elif fix_rate > 0.5:
                paras.append(
                    'The high fix rate indicates the terrain is rich with matchable features '
                    'and the filter is receiving frequent position updates, which should yield '
                    'low drift between fixes.'
                )
    sections.append(('4.  TERCOM Match Quality & Acceptance Rate', paras))

    # 5. Recommendations
    paras = []
    recs = []
    if not np.isnan(s['h_rms']) and s['h_rms'] > 50:
        recs.append('Horizontal RMS > 50 m: inspect the dynamic search radius and DEM resolution; '
                    'a larger search window may be needed if the UAV speed is high.')
    if not np.isnan(s['nis_consistent_frac']) and s['nis_consistent_frac'] < 0.80:
        recs.append('Poor NIS consistency: re-tune Q (process noise) and R (measurement noise) '
                    'matrices. Use the NIS mean value to determine the direction of the imbalance.')
    for ax_label, key in [('X', 'within3sigma_x'), ('Y', 'within3sigma_y'), ('Z', 'within3sigma_z')]:
        v = s.get(key, float('nan'))
        if not np.isnan(v) and v < 0.99:
            recs.append(f'Covariance {ax_label}-axis below 99%: the filter underestimates '
                        f'uncertainty — increase Q_{ax_label.lower()} or enable adaptive '
                        f'noise inflation on the ESKF.')
    if not np.isnan(s['tercom_fix_rate_hz']) and s['tercom_fix_rate_hz'] < 0.02:
        recs.append('Very low TERCOM fix rate: consider reducing profile_spacing_m or '
                    'increasing the sampling window to collect more terrain profiles per match.')
    if not recs:
        recs.append('No critical issues detected. The filter performance meets the expected '
                    'thresholds. For further improvement, consider A/B testing with tighter '
                    'Q/R values or a higher-resolution DEM.')
    paras.extend(f'• {r}' for r in recs)
    sections.append(('5.  Recommendations', paras))

    return sections


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Analyze TERCOM navigation CSV log and generate figures + PDF report.')
    parser.add_argument('csv', help='Path to tercom_log_*.csv file')
    parser.add_argument('--outdir', default=None,
                        help='Output directory (default: same directory as CSV)')
    parser.add_argument('--formats', default='png',
                        help='Comma-separated figure formats (default: png)')
    parser.add_argument('--no-report', action='store_true',
                        help='Skip PDF report generation')
    parser.add_argument('--origin-lat', type=float, default=None,
                        help='Latitude of ENU frame origin (deg) — enables '
                             'satellite-map trajectory plot (01b_trajectory_satellite)')
    parser.add_argument('--origin-lon', type=float, default=None,
                        help='Longitude of ENU frame origin (deg) — enables '
                             'satellite-map trajectory plot (01b_trajectory_satellite)')
    parser.add_argument('--satellite-zoom', type=int, default=17,
                        help='Tile zoom level for satellite map (1-19, default: 17). '
                             'Higher = more detail but more tiles to download. '
                             'Use 15-16 for long flights, 17-18 for short/local flights.')
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f'ERROR: File not found: {args.csv}')
        sys.exit(1)

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(outdir, exist_ok=True)

    OUTPUT_FORMATS.clear()
    OUTPUT_FORMATS.extend(f.strip() for f in args.formats.split(','))

    print(f'Loading {args.csv} ...')
    df = load_csv(args.csv)
    print(f'  Rows: {len(df):,}   Columns: {len(df.columns)}')

    generators = [
        fig_trajectory_xy,
        functools.partial(fig_trajectory_satellite,
                          origin_lat=args.origin_lat,
                          origin_lon=args.origin_lon,
                          satellite_zoom=args.satellite_zoom),
        fig_position_error_time,
        fig_error_statistics,
        fig_covariance_consistency,
        fig_nis_time,
        fig_tercom_quality,
        fig_speed_profile,
        fig_filter_state_timeline,
        fig_error_histogram,
        fig_cov_vs_error,
        fig_tercom_mad_vs_error,
        fig_trajectory_3d,
        fig_accepted_fixes_rate,
        fig_health_metrics,
        fig_summary_dashboard,
    ]

    print(f'\nGenerating {len(generators)} figures ...')
    for gen in generators:
        func_name = getattr(gen, '__name__', None) or getattr(gen, 'func', gen).__name__
        name = func_name.replace('fig_', '')
        print(f'  {name}')
        try:
            gen(df, outdir)
        except Exception as exc:
            print(f'    WARNING: failed — {exc}')

    if not args.no_report:
        print('\nComputing statistics ...')
        stats = compute_stats(df)
        print('Generating PDF report ...')
        try:
            generate_pdf_report(df, stats, args.csv, outdir,
                                origin_lat=args.origin_lat,
                                origin_lon=args.origin_lon,
                                satellite_zoom=args.satellite_zoom)
        except Exception as exc:
            print(f'  WARNING: PDF report failed — {exc}')

    print(f'\nDone. Output directory: {outdir}')


if __name__ == '__main__':
    main()
