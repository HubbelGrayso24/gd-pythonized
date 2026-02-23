from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
import random
import shutil
import warnings

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.io as sio

from main import run_model


REPLICATION_PARMS = {
    "M1_Upsilon_offdiag": "ORIGINAL",
    "M2_Symmetrize": "ORIGINAL",
    "M3_Upsilon_diag": "ORIGINAL",
    "M4_Compute_Dist": "ORIGINAL",
    "M5_ForWithinCellDist": "ORIGINAL",
    "M6_SanityCheckWithinCell": "ORIGINAL",
}


def _backup_if_exists(path: Path) -> None:
    if not path.exists():
        return

    stem = path.name
    parent = path.parent
    while True:
        suffix = random.randint(1, 10000)
        backup = parent / f"{stem}_{suffix}"
        if not backup.exists():
            shutil.move(str(path), str(backup))
            return


def create_storefolder(folder_name: str) -> Path:
    path = Path(folder_name)
    _backup_if_exists(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def move_storefolder(previousfolder: str, foldernewname: str) -> None:
    src = Path(previousfolder)
    if not src.exists():
        return

    dst = Path(foldernewname)
    _backup_if_exists(dst)
    shutil.move(str(src), str(dst))


def print_table_to_tex(table: np.ndarray, row_names: list[str], col_names: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l" + "c" * table.shape[1] + "}\n")
        f.write("\\hline\n")
        f.write(" & ".join(col_names) + " \\\\\n")
        f.write("\\hline\n")
        for i, row in enumerate(table):
            row_vals = " & ".join(f"{x:.4f}" for x in row)
            f.write(f"{row_names[i]} & {row_vals} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")


def combine_maps_panel(readfolder: Path, graphicsstorefilename: Path, files: list[str], rows: int, cols: int) -> None:
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.asarray(axes).reshape(-1)

    for i, fname in enumerate(files):
        ax = axes[i]
        img_path = readfolder / fname
        if img_path.exists():
            ax.imshow(mpimg.imread(img_path))
            ax.set_title(fname)
        else:
            ax.text(0.5, 0.5, f"Missing\n{fname}", ha="center", va="center")
        ax.axis("off")

    for j in range(len(files), len(axes)):
        axes[j].axis("off")

    graphicsstorefilename.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(graphicsstorefilename) + ".png", dpi=300)
    plt.close(fig)


def _period_stats(run_data: dict[str, np.ndarray], alpha: float, theta: float) -> dict[str, np.ndarray]:
    realgdp = run_data["realgdp"]
    tau = run_data["tau"]
    l = run_data["l"]
    realgdp_w = run_data["realgdp_w"]
    prod_w = run_data["prod_w"]
    u_w = run_data["u_w"]
    u2_w = run_data["u2_w"]

    T = int(realgdp.shape[1])
    corr_rgdppop = np.zeros(T)
    corr_prpop = np.zeros(T)
    corr_prrgdp = np.zeros(T)

    for t in range(T):
        rgdp_vector = np.log(np.maximum(realgdp[:, t], 1e-300))
        pop_vector = np.log(np.maximum(l[:, t], 1e-300))
        pr_vector = np.log(np.maximum((tau[:, t] * l[:, t] ** alpha) ** (1 / theta), 1e-300))

        corr_rgdppop[t] = np.corrcoef(rgdp_vector, pop_vector)[0, 1]
        corr_prpop[t] = np.corrcoef(pr_vector, pop_vector)[0, 1]
        corr_prrgdp[t] = np.corrcoef(pr_vector, rgdp_vector)[0, 1]

    prgrowth = np.ones(T)
    rgdpgrowth = np.ones(T)
    ugrowth = np.ones(T)
    u2growth = np.ones(T)
    prgrowth[1:] = prod_w[1:] / np.maximum(prod_w[:-1], 1e-300)
    rgdpgrowth[1:] = realgdp_w[1:] / np.maximum(realgdp_w[:-1], 1e-300)
    ugrowth[1:] = u_w[1:] / np.maximum(u_w[:-1], 1e-300)
    u2growth[1:] = u2_w[1:] / np.maximum(u2_w[:-1], 1e-300)

    return {
        "corr_rgdppop": corr_rgdppop,
        "corr_prpop": corr_prpop,
        "corr_prrgdp": corr_prrgdp,
        "prgrowth": prgrowth,
        "rgdpgrowth": rgdpgrowth,
        "ugrowth": ugrowth,
        "u2growth": u2growth,
        "logprworld": np.log(np.maximum(prod_w, 1e-300)),
        "logrgdpworld": np.log(np.maximum(realgdp_w, 1e-300)),
        "loguworld": np.log(np.maximum(u_w, 1e-300)),
        "logu2world": np.log(np.maximum(u2_w, 1e-300)),
    }


def _plot_figure_4(plot_1000: dict[str, np.ndarray], plot_375: dict[str, np.ndarray], plot_0: dict[str, np.ndarray], out_path: Path) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    axs[0].plot(plot_1000["corr_rgdppop"], "-")
    axs[0].plot(plot_375["corr_rgdppop"], "--")
    axs[0].plot(plot_0["corr_rgdppop"], "-.")
    axs[0].set_title("Corr (log real GDP per capita, log population density)")

    axs[1].plot(plot_1000["corr_prpop"], "-")
    axs[1].plot(plot_375["corr_prpop"], "--")
    axs[1].plot(plot_0["corr_prpop"], "-.")
    axs[1].set_title("Corr (log productivity, log population density)")

    axs[2].plot(plot_1000["corr_prrgdp"], "-")
    axs[2].plot(plot_375["corr_prrgdp"], "--")
    axs[2].plot(plot_0["corr_prrgdp"], "-.")
    axs[2].set_title("Corr (log productivity, log real GDP per capita)")

    for ax in axs:
        ax.set_xlabel("Time")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def run_a00_main() -> None:
    projectfolder = Path.cwd()

    warnings.warn(
        "A0-A8 preprocessing stages are MATLAB-specific and are not run in this Python wrapper. "
        "This wrapper assumes Data/*.mat and Data/*.csv already exist.",
        stacklevel=1,
    )

    output_log = projectfolder / "output.txt"
    with output_log.open("w", encoding="utf-8") as diary:
        create_storefolder("Maps")
        create_storefolder("Output")

        with redirect_stdout(diary):
            benchmark_data = run_model(curledtheta=1.0)

    move_storefolder("Maps", "Maps_benchmark")
    move_storefolder("Output", "Output_benchmark")

    curledthetavec = [1, 0.875, 0.750, 0.500, 0.375, 0.250, 0.125, 0]
    colfields = ["PDV_realgdp_w", "PDV_u_w", "PDV_u2_w", "migr_ctry"]
    table3 = np.zeros((len(curledthetavec), len(colfields)))
    experiment_runs: dict[str, dict[str, np.ndarray]] = {}

    for row, curledtheta in enumerate(curledthetavec):
        print(f"Curl Theta Migration restriction = {curledtheta:.3f}")
        create_storefolder("Maps")
        create_storefolder("Output")

        run_data = run_model(curledtheta=curledtheta)

        for col, fname in enumerate(colfields):
            scalar = run_data[fname]
            if fname == "migr_ctry":
                scalar = float(np.asarray(scalar).reshape(-1)[0]) * 100
            table3[row, col] = float(np.asarray(scalar).reshape(-1)[0])

        exp_tag = int(round(curledtheta * 1000))
        experiment_runs[str(exp_tag)] = run_data
        move_storefolder("Maps", f"Maps_curledtheta_{exp_tag}")
        move_storefolder("Output", f"Output_curledtheta_{exp_tag}")

    table3_levels = table3.copy()
    benchmark = np.tile(table3[0, 0:3], (table3.shape[0], 1))
    table3[:, 0:3] = (table3[:, 0:3] - benchmark) / benchmark * 100

    table3rownames = ["1", ".875", ".750", ".500", ".375", ".250", ".125", "0"]
    table3colnames = ["Mobility", "Real Income (1)", "Welfare (2)", "Expected Welfare (3)", "Migration Flows (4)"]

    create_storefolder("Graphics")
    sio.savemat("Graphics/Table3_levels.mat", {"Table3_levels": table3_levels})
    sio.savemat("Graphics/Table3.mat", {"Table3": table3})
    print_table_to_tex(table3, table3rownames, table3colnames, Path("Graphics/table3.tex"))

    combine_maps_panel(
        Path("Maps_benchmark"),
        Path("Graphics/Fig2"),
        ["PD_NF_1_1000.png", "PR_NF_1_1000.png", "U_NF_1_1000.png", "RO_NF_1_1000.png"],
        2,
        2,
    )
    combine_maps_panel(
        Path("Maps_benchmark"),
        Path("Graphics/Fig3"),
        ["PD_NF_600_1000.png", "PR_NF_600_1000.png", "U_NF_600_1000.png", "RO_NF_600_1000.png"],
        2,
        2,
    )
    combine_maps_panel(
        Path("Maps_curledtheta_0"),
        Path("Graphics/Fig6"),
        ["PD_NF_1_1000.png", "PR_NF_1_1000.png", "U_NF_1_1000.png", "RO_NF_1_1000.png"],
        2,
        2,
    )
    combine_maps_panel(
        Path("Maps_curledtheta_0"),
        Path("Graphics/Fig7"),
        ["PD_NF_600_1000.png", "PR_NF_600_1000.png", "U_NF_600_1000.png", "RO_NF_600_1000.png"],
        2,
        2,
    )

    vars_ = benchmark_data["vars"]
    alpha = float(vars_[21])
    theta = float(vars_[22])

    plot_1000 = _period_stats(experiment_runs["1000"], alpha, theta)
    plot_375 = _period_stats(experiment_runs["375"], alpha, theta)
    plot_0 = _period_stats(experiment_runs["0"], alpha, theta)

    _plot_figure_4(plot_1000, plot_375, plot_0, Path("Graphics/Fig4.png"))

    warnings.warn(
        "Figures 1, 5, and 8 in MATLAB A00_main.m depend on helper functions not present in the pythonized repo "
        "(maps_period_0 / B3_output_read_fun / B3_worldaggregate_fun) and are intentionally not auto-generated.",
        stacklevel=1,
    )


if __name__ == "__main__":
    run_a00_main()
