"""CLI entry point for audioscope."""
from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    TextColumn,
    SpinnerColumn,
)
from rich.panel import Panel
from rich.table import Table
from rich import box

from audioscope.audio import AudioFile
from audioscope.themes import THEMES
from audioscope.renderer import build_modules, Renderer
from audioscope.modules import REGISTRY

console = Console()

AVAILABLE_MODULES = list(REGISTRY.keys())
AVAILABLE_THEMES = list(THEMES.keys())


def _print_banner() -> None:
    console.print(
        Panel.fit(
            "[bold magenta]AudioScope[/bold magenta]  [dim]v0.1.0[/dim]\n"
            "[dim]Audio visualization → PNG frame sequences[/dim]",
            border_style="bright_black",
        )
    )


def _print_session_info(
    audio: AudioFile,
    output: Path,
    modules: list,
    fps: float,
    height: int,
    theme_name: str,
    total_frames: int,
) -> None:
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column(style="dim", width=14)
    table.add_column()
    table.add_row("Input", str(audio.path))
    table.add_row("Channels", f"{'Stereo' if audio.is_stereo else 'Mono'}  ({audio.n_channels}ch, {audio.sample_rate} Hz)")
    table.add_row("Duration", f"{audio.duration:.2f}s")
    table.add_row("Output", str(output))
    table.add_row("Modules", ", ".join(m.name for m in modules))
    table.add_row("FPS", str(fps))
    table.add_row("Height", f"{height}px")
    table.add_row("Theme", theme_name)
    table.add_row("Frames", str(total_frames))
    console.print(table)


@click.command(name="visualize")
@click.option("-i", "--input", "input_path", required=True,
              type=click.Path(exists=True, dir_okay=False, readable=True),
              help="Input WAV file.")
@click.option("-o", "--output", "output_path", required=True,
              type=click.Path(file_okay=False),
              help="Output directory for PNG frame sequence.")
@click.option("-m", "--modules", "module_list", default="spectrum,waveform,spectrogram",
              show_default=True,
              help=(
                  f"Comma-separated list of modules. "
                  f"Available: {', '.join(AVAILABLE_MODULES)}"
              ))
@click.option("--fps", default=24.0, show_default=True, type=float,
              help="Frames per second.")
@click.option("--height", default=400, show_default=True, type=int,
              help="Height of each module in pixels.")
@click.option("--width", "width_override", default=None, type=int,
              help="Force a fixed width for every module (overrides aspect ratios).")
@click.option("--theme", default="neon", show_default=True,
              type=click.Choice(AVAILABLE_THEMES, case_sensitive=False),
              help="Color theme.")
@click.option("--fft-size", default=4096, show_default=True, type=int,
              help="FFT window size (power of 2 recommended).")
@click.option("--window", "window_func", default="hann", show_default=True,
              type=click.Choice(["hann", "hamming", "blackman", "bartlett", "flat"],
                                case_sensitive=False),
              help="FFT window function.")
@click.option("--db-range", default=80.0, show_default=True, type=float,
              help="Dynamic range in dB (height of the dB scale).")
@click.option("--min-freq", default=20.0, show_default=True, type=float,
              help="Minimum displayed frequency in Hz.")
@click.option("--max-freq", default=20000.0, show_default=True, type=float,
              help="Maximum displayed frequency in Hz.")
@click.option("--smoothing", default=0.72, show_default=True,
              type=click.FloatRange(0.0, 0.99),
              help="Spectrum smoothing factor (0=off, 0.95=heavy).")
@click.option("--peak-hold", default=2.0, show_default=True, type=float,
              help="Peak-hold time in seconds.")
@click.option("--window-secs", default=0.5, show_default=True, type=float,
              help="Audio window fed to each module per frame (seconds).")
@click.option("--gain", default=0.0, show_default=True, type=float,
              help="Input gain in dB (applied before all processing).")
@click.option("--normalize/--no-normalize", default=False,
              help="Normalize audio to 0 dBFS peak before processing.")
@click.option("--no-labels/--labels", default=False,
              help="Disable text labels on all modules.")
@click.option("--no-peaks/--peaks", default=False,
              help="Disable peak-hold markers on spectrum.")
@click.option("--stereo-waveform/--mono-waveform", default=True, show_default=True,
              help="Show both channels in waveform module.")
@click.option("--format", "img_format", default="png", show_default=True,
              type=click.Choice(["png", "jpg"], case_sensitive=False),
              help="Output image format.")
@click.option("--quality", default=95, show_default=True, type=int,
              help="JPEG quality (only used with --format jpg).")
@click.option("--separator", default=2, show_default=True, type=int,
              help="Pixel gap between modules.")
@click.option("--n-slices", default=48, show_default=True, type=int,
              help="Number of history slices in 3D spectrogram.")
def main(
    input_path: str,
    output_path: str,
    module_list: str,
    fps: float,
    height: int,
    width_override: Optional[int],
    theme: str,
    fft_size: int,
    window_func: str,
    db_range: float,
    min_freq: float,
    max_freq: float,
    smoothing: float,
    peak_hold: float,
    window_secs: float,
    gain: float,
    normalize: bool,
    no_labels: bool,
    no_peaks: bool,
    stereo_waveform: bool,
    img_format: str,
    quality: int,
    separator: int,
    n_slices: int,
) -> None:
    """Render an audio file as a PNG frame sequence."""
    _print_banner()

    # ------------------------------------------------------------------ load audio
    try:
        audio = AudioFile(input_path)
    except Exception as exc:
        console.print(f"[red]Error loading audio:[/red] {exc}")
        sys.exit(1)

    if normalize:
        audio.normalize()
    if gain != 0.0:
        audio.apply_gain_db(gain)

    # ------------------------------------------------------------------ parse modules
    names = [m.strip().lower() for m in module_list.split(",") if m.strip()]
    if not names:
        console.print("[red]No modules specified.[/red]")
        sys.exit(1)

    for n in names:
        if n not in REGISTRY:
            console.print(f"[red]Unknown module '{n}'.[/red] Available: {', '.join(AVAILABLE_MODULES)}")
            sys.exit(1)

    # ------------------------------------------------------------------ build shared config
    shared_cfg = {
        "fft_size": fft_size,
        "window_func": window_func,
        "db_range": db_range,
        "min_freq": min_freq,
        "max_freq": max_freq,
        "show_labels": not no_labels,
    }
    module_configs = {
        "spectrum": {
            **shared_cfg,
            "smoothing": smoothing,
            "peak_hold_secs": peak_hold,
            "show_peaks": not no_peaks,
        },
        "waveform": {
            "duration_secs": window_secs,
            "stereo": stereo_waveform,
            "show_labels": not no_labels,
        },
        "spectrogram": {**shared_cfg},
        "stereometer": {
            "show_labels": not no_labels,
            "db_range": db_range,
        },
        "spectrogram3d": {
            **shared_cfg,
            "n_slices": n_slices,
        },
    }

    selected_configs = {n: module_configs.get(n, {}) for n in names}
    color_theme = THEMES[theme]

    try:
        modules = build_modules(names, height, color_theme, width_override, selected_configs)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    output_dir = Path(output_path)
    renderer = Renderer(audio, modules, fps, height, window_secs, separator)

    _print_session_info(audio, output_dir, modules, fps, height, theme, renderer.total_frames)

    # ------------------------------------------------------------------ render
    console.print("[bold]Rendering frames...[/bold]")
    t_start = time.time()

    with Progress(
        SpinnerColumn(),
        BarColumn(bar_width=40, complete_style="magenta", finished_style="bright_magenta"),
        TaskProgressColumn(),
        TextColumn("[dim]{task.completed}/{task.total} frames[/dim]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("render", total=renderer.total_frames)

        def on_progress(done: int, total: int) -> None:
            progress.update(task, completed=done)

        renderer.render_all(
            output_dir,
            fmt=img_format,
            quality=quality,
            progress_callback=on_progress,
        )

    elapsed = time.time() - t_start
    total_w = sum(m.width for m in modules) + separator * (len(modules) - 1)
    console.print(
        f"\n[bold green]Done![/bold green] "
        f"{renderer.total_frames} frames "
        f"([dim]{total_w}×{height}px[/dim]) "
        f"saved to [cyan]{output_dir}[/cyan] "
        f"in [dim]{elapsed:.1f}s[/dim] "
        f"([dim]{renderer.total_frames / elapsed:.1f} fps[/dim])"
    )
